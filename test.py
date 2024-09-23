#!/usr/bin/env python

import argparse
import glob
import os
import math
import stat
import numpy as np
import warp as wp
import signal
import threading

from ctypes import *
from dataclasses import dataclass
from typing import Optional, Tuple

from contextlib import ExitStack
from pathlib import Path
from libevdev import *
from signal import pthread_sigmask, pthread_kill, sigwait
from threading import main_thread, Thread

import shaderbang.input
from shaderbang.input import TouchSlot
from shaderbang.inotify import INotify, IN_CREATE, IN_ATTRIB
from lib import glsl, options

#os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ['USE_ACCELERATE'] = 'False'
#os.environ['ERROR_CHECKING'] = 'False'
from OpenGL import setPlatform
setPlatform('egl')

from OpenGL.GL import *
from OpenGL.GLU import *

wp.init()

numSubsteps = 60
timeStep = 1.0 / 30.0
gravity = wp.vec3(0.0, -10.0, 0.0)
paused = False
hidden = False

# 0 Coloring
# 1 Jacobi
solveType = 0
jacobiScale = 0.2

clothNumX = 500
clothNumY = 500
clothY = 2.2
clothSpacing = 0.01
sphereCenter = wp.vec3(0.0, 1.5, 0.0)
sphereRadius = 0.5


@dataclass
class Particle:
    id: int
    mass: float
    depth: float


class Cloth:

    @wp.kernel
    def computeRestLengths(
            pos: wp.array(dtype=wp.vec3),
            constIds: wp.array(dtype=wp.int32),
            restLengths: wp.array(dtype=float)):
        cNr = wp.tid()
        p0 = pos[constIds[2 * cNr]]
        p1 = pos[constIds[2 * cNr + 1]]
        restLengths[cNr] = wp.length(p1 - p0)

    # -----------------------------------------------------
    def __init__(self, yOffset, numX, numY, spacing, sphereCenter, sphereRadius):
        self.renderParticles = []

        self.sphereCenter = sphereCenter
        self.sphereRadius = sphereRadius

        if numX % 2 == 1:
            numX = numX + 1
        if numY % 2 == 1:
            numY = numY + 1

        self.spacing = spacing
        self.numParticles = (numX + 1) * (numY + 1)
        pos = np.zeros(3 * self.numParticles)
        normals = np.zeros(3 * self.numParticles)
        invMass = np.zeros(self.numParticles)

        for xi in range(numX + 1):
            for yi in range(numY + 1):
                id = xi * (numY + 1) + yi
                pos[3 * id] = (-numX * 0.5 + xi) * spacing
                pos[3 * id + 1] = yOffset
                pos[3 * id + 2] = (-numY * 0.5 + yi) * spacing
                invMass[id] = 1.0

        self.pos = wp.array(pos, dtype=wp.vec3, device="cuda")
        self.prevPos = wp.array(pos, dtype=wp.vec3, device="cuda")
        self.restPos = wp.array(pos, dtype=wp.vec3, device="cuda")
        self.invMass = wp.array(invMass, dtype=float, device="cuda")
        self.corr = wp.array(np.zeros(3 * self.numParticles), dtype=wp.vec3, device="cuda")
        self.vel = wp.array(np.zeros(3 * self.numParticles), dtype=wp.vec3, device="cuda")
        self.normals = wp.array(normals, dtype=wp.vec3, device="cuda")

        self.hostInvMass = wp.array(invMass, dtype=float, device="cpu")
        self.hostPos = wp.array(pos, dtype=wp.vec3, device="cpu")
        self.hostNormals = wp.array(normals, dtype=wp.vec3, device="cpu")

        # constraints

        self.passSizes = [
            (numX + 1) * math.floor(numY / 2),
            (numX + 1) * math.floor(numY / 2),
            math.floor(numX / 2) * (numY + 1),
            math.floor(numX / 2) * (numY + 1),
            2 * numX * numY + (numX + 1) * (numY - 1) + (numY + 1) * (numX - 1)
        ]
        self.passIndependent = [
            True, True, True, True, False
        ]

        self.numDistConstraints = 0
        for passSize in self.passSizes:
            self.numDistConstraints = self.numDistConstraints + passSize

        distConstIds = np.zeros(2 * self.numDistConstraints, dtype=wp.int32)

        # stretch constraints

        i = 0
        for passNr in range(2):
            for xi in range(numX + 1):
                for yi in range(math.floor(numY / 2)):
                    distConstIds[2 * i] = xi * (numY + 1) + 2 * yi + passNr
                    distConstIds[2 * i + 1] = xi * (numY + 1) + 2 * yi + passNr + 1
                    i = i + 1

        for passNr in range(2):
            for xi in range(math.floor(numX / 2)):
                for yi in range(numY + 1):
                    distConstIds[2 * i] = (2 * xi + passNr) * (numY + 1) + yi
                    distConstIds[2 * i + 1] = (2 * xi + passNr + 1) * (numY + 1) + yi
                    i = i + 1

        # shear constraints

        for xi in range(numX):
            for yi in range(numY):
                distConstIds[2 * i] = xi * (numY + 1) + yi
                distConstIds[2 * i + 1] = (xi + 1) * (numY + 1) + yi + 1
                i = i + 1
                distConstIds[2 * i] = (xi + 1) * (numY + 1) + yi
                distConstIds[2 * i + 1] = xi * (numY + 1) + yi + 1
                i = i + 1

        # bending constraints

        for xi in range(numX + 1):
            for yi in range(numY - 1):
                distConstIds[2 * i] = xi * (numY + 1) + yi
                distConstIds[2 * i + 1] = xi * (numY + 1) + yi + 2
                i = i + 1

        for xi in range(numX - 1):
            for yi in range(numY + 1):
                distConstIds[2 * i] = xi * (numY + 1) + yi
                distConstIds[2 * i + 1] = (xi + 2) * (numY + 1) + yi
                i = i + 1

        self.distConstIds = wp.array(distConstIds, dtype=wp.int32, device="cuda")
        self.constRestLengths = wp.zeros(self.numDistConstraints, dtype=float, device="cuda")

        wp.launch(kernel=self.computeRestLengths,
                  inputs=[self.pos, self.distConstIds, self.constRestLengths],
                  dim=self.numDistConstraints,  device="cuda")

        # tri ids

        self.numTris = 2 * numX * numY
        self.hostTriIds = np.zeros(3 * self.numTris, dtype = np.int32)

        i = 0
        for xi in range(numX):
            for yi in range(numY):
                id0 = xi * (numY + 1) + yi
                id1 = (xi + 1) * (numY + 1) + yi
                id2 = (xi + 1) * (numY + 1) + yi + 1
                id3 = xi * (numY + 1) + yi + 1

                self.hostTriIds[i] = id0
                self.hostTriIds[i + 1] = id1
                self.hostTriIds[i + 2] = id2

                self.hostTriIds[i + 3] = id0
                self.hostTriIds[i + 4] = id2
                self.hostTriIds[i + 5] = id3

                i = i + 6

        self.triIds = wp.array(self.hostTriIds, dtype=wp.int32, device="cuda")

        self.triDist = wp.zeros(self.numTris, dtype=float, device="cuda")
        self.hostTriDist = wp.zeros(self.numTris, dtype=float, device="cpu")

        print(str(self.numTris) + " triangles created")
        print(str(self.numDistConstraints) + " distance constraints created")
        print(str(self.numParticles) + " particles created")

    # ----------------------------------
    @wp.kernel
    def addNormals(
            pos: wp.array(dtype=wp.vec3),
            triIds: wp.array(dtype=wp.int32),
            normals: wp.array(dtype=wp.vec3)):
        triNr = wp.tid()

        id0 = triIds[3 * triNr]
        id1 = triIds[3 * triNr + 1]
        id2 = triIds[3 * triNr + 2]
        normal = wp.cross(pos[id1] - pos[id0], pos[id2] - pos[id0])
        wp.atomic_add(normals, id0, normal)
        wp.atomic_add(normals, id1, normal)
        wp.atomic_add(normals, id2, normal)

    @wp.kernel
    def normalizeNormals(normals: wp.array(dtype=wp.vec3)):
        pNr = wp.tid()
        normals[pNr] = wp.normalize(normals[pNr])

    def updateMesh(self):
        self.normals.zero_()
        wp.launch(kernel=self.addNormals, inputs=[self.pos, self.triIds, self.normals], dim=self.numTris, device="cuda")
        wp.launch(kernel=self.normalizeNormals, inputs=[self.normals], dim=self.numParticles, device="cuda")
        wp.copy(self.hostNormals, self.normals)

    # ----------------------------------

    @wp.kernel
    def integrate(
            dt: float,
            gravity: wp.vec3,
            invMass: wp.array(dtype=float),
            prevPos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3),
            sphereCenter: wp.vec3,
            sphereRadius: float):

        pNr = wp.tid()

        prevPos[pNr] = pos[pNr]
        if invMass[pNr] == 0.0:
            return
        vel[pNr] = vel[pNr] + gravity * dt
        pos[pNr] = pos[pNr] + vel[pNr] * dt

        # collisions
        thickness = 0.001
        friction = 0.01

        d = wp.length(pos[pNr] - sphereCenter)
        if d < (sphereRadius + thickness):
            p = pos[pNr] * (1.0 - friction) + prevPos[pNr] * friction
            r = p - sphereCenter
            d = wp.length(r)
            pos[pNr] = sphereCenter + r * ((sphereRadius + thickness) / d)

        p = pos[pNr]
        if p[1] < thickness:
            p = pos[pNr] * (1.0 - friction) + prevPos[pNr] * friction
            pos[pNr] = wp.vec3(p[0], thickness, p[2])

    # ----------------------------------
    @wp.kernel
    def solveDistanceConstraints(
            solveType: wp.int32,
            firstConstraint: wp.int32,
            invMass: wp.array(dtype=float),
            pos: wp.array(dtype=wp.vec3),
            corr: wp.array(dtype=wp.vec3),
            constIds: wp.array(dtype=wp.int32),
            restLengths: wp.array(dtype=float)):

        cNr = firstConstraint + wp.tid()
        id0 = constIds[2 * cNr]
        id1 = constIds[2 * cNr + 1]
        w0 = invMass[id0]
        w1 = invMass[id1]
        w = w0 + w1
        if w == 0.0:
            return
        p0 = pos[id0]
        p1 = pos[id1]
        d = p1 - p0
        n = wp.normalize(d)
        l = wp.length(d)
        l0 = restLengths[cNr]
        dP = n * (l - l0) / w
        if solveType == 1:
            wp.atomic_add(corr, id0, w0 * dP)
            wp.atomic_sub(corr, id1, w1 * dP)
        else:
            wp.atomic_add(pos, id0, w0 * dP)
            wp.atomic_sub(pos, id1, w1 * dP)

    # ----------------------------------
    @wp.kernel
    def addCorrections(
            pos: wp.array(dtype=wp.vec3),
            corr: wp.array(dtype=wp.vec3),
            scale: float):
        pNr = wp.tid()
        pos[pNr] = pos[pNr] + corr[pNr] * scale

    # ----------------------------------
    @wp.kernel
    def updateVel(
            dt: float,
            prevPos: wp.array(dtype=wp.vec3),
            pos: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3)):
        pNr = wp.tid()
        vel[pNr] = (pos[pNr] - prevPos[pNr]) / dt

    # ----------------------------------
    def simulate(self):
        dt = timeStep / numSubsteps
        numPasses = len(self.passSizes)

        for step in range(numSubsteps):
            wp.launch(kernel=self.integrate,
                      inputs=[dt, gravity, self.invMass, self.prevPos, self.pos, self.vel, self.sphereCenter, self.sphereRadius],
                      dim=self.numParticles, device="cuda")

            if solveType == 0:
                firstConstraint = 0
                for passNr in range(numPasses):
                    numConstraints = self.passSizes[passNr]

                    if self.passIndependent[passNr]:
                        wp.launch(kernel=self.solveDistanceConstraints,
                                  inputs=[0, firstConstraint, self.invMass, self.pos, self.corr, self.distConstIds, self.constRestLengths],
                                  dim=numConstraints,  device = "cuda")
                    else:
                        self.corr.zero_()
                        wp.launch(kernel=self.solveDistanceConstraints,
                                  inputs=[1, firstConstraint, self.invMass, self.pos, self.corr, self.distConstIds, self.constRestLengths],
                                  dim=numConstraints, device="cuda")
                        wp.launch(kernel=self.addCorrections,
                                  inputs=[self.pos, self.corr, jacobiScale],
                                  dim=self.numParticles, device="cuda")

                    firstConstraint = firstConstraint + numConstraints

            elif solveType == 1:
                self.corr.zero_()
                wp.launch(kernel=self.solveDistanceConstraints,
                          inputs=[1, 0, self.invMass, self.pos, self.corr, self.distConstIds, self.constRestLengths],
                          dim=self.numDistConstraints, device="cuda")
                wp.launch(kernel=self.addCorrections,
                          inputs=[self.pos, self.corr, jacobiScale],
                          dim=self.numParticles, device="cuda")

            wp.launch(kernel=self.updateVel,
                      inputs=[dt, self.prevPos, self.pos, self.vel], dim=self.numParticles, device="cuda")

        wp.copy(self.hostPos, self.pos)

    # -------------------------------------------------
    def reset(self):
        self.vel.zero_()
        wp.copy(self.pos, self.restPos)

    # -------------------------------------------------
    @wp.kernel
    def raycast_triangle(
            orig: wp.vec3,
            dir: wp.vec3,
            pos: wp.array(dtype=wp.vec3),
            triIds: wp.array(dtype=wp.int32),
            dist: wp.array(dtype=float)):
        triNr = wp.tid()
        noHit = 1.0e6

        id0 = triIds[3 * triNr]
        id1 = triIds[3 * triNr + 1]
        id2 = triIds[3 * triNr + 2]
        # pNr = wp.tid()

        edge1 = pos[id1] - pos[id0]
        edge2 = pos[id2] - pos[id0]
        pvec = wp.cross(dir, edge2)
        det = wp.dot(edge1, pvec)

        if det == 0.0:
            dist[triNr] = noHit
            return

        inv_det = 1.0 / det
        tvec = orig - pos[id0]
        u = wp.dot(tvec, pvec) * inv_det
        if u < 0.0 or u > 1.0:
            dist[triNr] = noHit
            return

        qvec = wp.cross(tvec, edge1)
        v = wp.dot(dir, qvec) * inv_det
        if v < 0.0 or u + v > 1.0:
            dist[triNr] = noHit
            return

        dist[triNr] = wp.dot(edge2, qvec) * inv_det

    # ------------------------------------------------
    def start_drag(self, orig: wp.vec3, dir: wp.vec3) -> Optional[Particle]:
        wp.launch(kernel=self.raycast_triangle, inputs=[
            wp.vec3(orig[0], orig[1], orig[2]), wp.vec3(dir[0], dir[1], dir[2]),
            self.pos, self.triIds, self.triDist], dim=self.numTris, device="cuda")
        wp.copy(self.hostTriDist, self.triDist)

        dists = self.hostTriDist.numpy()
        min_dist_id = np.argmin(dists)
        if dists[min_dist_id] >= 1.0e6:
            return None

        particle_id = self.hostTriIds[3 * min_dist_id]
        drag_depth = dists[min_dist_id]

        inv_mass = self.hostInvMass.numpy()
        drag_inv_mass = inv_mass[particle_id]
        inv_mass[particle_id] = 0.0
        wp.copy(self.invMass, self.hostInvMass)

        pos = self.hostPos.numpy()
        pos[particle_id] = wp.add(orig, wp.mul(dir, drag_depth))
        wp.copy(self.pos, self.hostPos)

        self.renderParticles.append(particle_id)

        return Particle(id=particle_id.item(), mass=drag_inv_mass, depth=drag_depth)

    def drag(self, orig: wp.vec3, dir: wp.vec3, particle: Particle):
        pos = self.hostPos.numpy()
        pos[particle.id] = wp.add(orig, wp.mul(dir, particle.depth))
        wp.copy(self.pos, self.hostPos)

    def end_drag(self, particle: Particle):
        invMass = self.hostInvMass.numpy()
        invMass[particle.id] = particle.mass
        wp.copy(self.invMass, self.hostInvMass)
        self.renderParticles.remove(particle.id)

    def render(self):
        # cloth
        two_colors = False

        glColor3f(1.0, 0.0, 0.0)
        glNormal3f(0.0, 0.0, -1.0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, self.hostPos.numpy())
        glNormalPointer(GL_FLOAT, 0, self.hostNormals.numpy())

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if two_colors:
            glCullFace(GL_FRONT)
            glColor3f(1.0, 1.0, 0.0)
            glDrawElementsui(GL_TRIANGLES, self.hostTriIds)
            glCullFace(GL_BACK)
            glColor3f(1.0, 0.0, 0.0)
            glDrawElementsui(GL_TRIANGLES, self.hostTriIds)
        else:
            glDisable(GL_CULL_FACE)
            glColor3f(1.0, 0.0, 0.0)
            glDrawElementsui(GL_TRIANGLES, self.hostTriIds)
            glEnable(GL_CULL_FACE)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

        # kinematic particles
        glColor3f(1.0, 1.0, 1.0)
        pos = self.hostPos.numpy()

        q = gluNewQuadric()

        for particle_id in self.renderParticles:
            glPushMatrix()
            p = pos[particle_id]
            glTranslatef(p[0], p[1], p[2])
            gluSphere(q, 0.02, 40, 40)
            glPopMatrix()

        # sphere
        glColor3f(0.8, 0.8, 0.8)

        glPushMatrix()
        glTranslatef(self.sphereCenter[0], self.sphereCenter[1], self.sphereCenter[2])
        gluSphere(q, self.sphereRadius, 40, 40)
        glPopMatrix()

        gluDeleteQuadric(q)


class Camera:
    def __init__(self):
        self.pos = wp.vec3(0.0, 1.0, 5.0)
        self.forward = wp.vec3(0.0, 0.0, -1.0)
        self.up = wp.vec3(0.0, 1.0, 0.0)
        self.right = wp.cross(self.forward, self.up)

    def set_view(self):
        viewport = glGetIntegerv(GL_VIEWPORT)
        width = viewport[2] - viewport[0]
        height = viewport[3] - viewport[1]

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        gluPerspective(40.0, float(width) / float(height), 0.01, 1000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        gluLookAt(
            self.pos[0], self.pos[1], self.pos[2],
            self.pos[0] + self.forward[0], self.pos[1] + self.forward[1], self.pos[2] + self.forward[2],
            self.up[0], self.up[1], self.up[2])

    def lookAt(self, pos, at):
        self.pos = pos
        self.forward = wp.sub(at, pos)
        self.forward = wp.normalize(self.forward)
        self.up = wp.vec3(0.0, 1.0, 0.0)
        self.right = wp.cross(self.forward, self.up)
        self.right = wp.normalize(self.right)
        self.up = wp.cross(self.right, self.forward)

    def rot(self, unitAxis, angle, v):
        q = wp.quat_from_axis_angle(unitAxis, angle)
        return wp.quat_rotate(q, v)

    def handleMouseOrbit(self, dx, dy, center):
        offset = wp.sub(self.pos, center)
        offset = [
            wp.dot(self.right, offset),
            wp.dot(self.forward, offset),
            wp.dot(self.up, offset)]

        scale = 0.01
        self.forward = self.rot(self.up, -dx * scale, self.forward)
        self.forward = self.rot(self.right, -dy * scale, self.forward)
        self.up = self.rot(self.up, -dx * scale, self.up)
        self.up = self.rot(self.right, -dy * scale, self.up)

        self.right = wp.cross(self.forward, self.up)
        self.right = wp.vec3(self.right[0], 0.0, self.right[2])
        self.right = wp.normalize(self.right)
        self.up = wp.cross(self.right, self.forward)
        self.up = wp.normalize(self.up)
        self.forward = wp.cross(self.up, self.right)
        self.pos = wp.add(center, wp.mul(self.right, offset[0]))
        self.pos = wp.add(self.pos, wp.mul(self.forward, offset[1]))
        self.pos = wp.add(self.pos, wp.mul(self.up, offset[2]))


camera = Camera()
cloth = Cloth(clothY, clothNumX, clothNumY, clothSpacing, sphereCenter, sphereRadius)

groundNumTiles = 30
groundTileSize = 0.5

groundVerts = np.zeros(3 * 4 * groundNumTiles * groundNumTiles, dtype=float)
groundColors = np.zeros(3 * 4 * groundNumTiles * groundNumTiles, dtype=float)

squareVerts = [[0, 0], [0, 1], [1, 1], [1, 0]]
r = groundNumTiles / 2.0 * groundTileSize

for xi in range(groundNumTiles):
    for zi in range(groundNumTiles):
        x = (-groundNumTiles / 2.0 + xi) * groundTileSize
        z = (-groundNumTiles / 2.0 + zi) * groundTileSize
        p = xi * groundNumTiles + zi
        for i in range(4):
            q = 4 * p + i
            px = x + squareVerts[i][0] * groundTileSize
            pz = z + squareVerts[i][1] * groundTileSize
            groundVerts[3 * q] = px
            groundVerts[3 * q + 2] = pz
            col = 0.4
            if (xi + zi) % 2 == 1:
                col = 0.8
            pr = math.sqrt(px * px + pz * pz)
            d = max(0.0, 1.0 - pr / r)
            col = col * d
            for j in range(3):
                groundColors[3 * q + j] = col


@CFUNCTYPE(None, c_uint, c_uint)
def init(width, height):
    glViewport(0, 0, width, height)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_CULL_FACE)
    glShadeModel(GL_SMOOTH)
    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    ambient_color = [0.2, 0.2, 0.2, 1.0]
    diffuse_color = [0.8, 0.8, 0.8, 1.0]
    specular_color = [1.0, 1.0, 1.0, 1.0]

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_color)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_color)
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular_color)

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular_color)
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)

    light_position = [10.0, 10.0, 10.0, 0.0]
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    glEnable(GL_NORMALIZE)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 1.0)


@CFUNCTYPE(None, c_uint64, c_float)
def render(frame, time):
    # camera
    # camera.handleMouseOrbit(0.5, 0, wp.vec3(0.0, 1.0, 0.0))
    camera.set_view()

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # ground plane
    glColor3f(1.0, 1.0, 1.0)
    glNormal3f(0.0, 1.0, 0.0)

    numVerts = math.floor(len(groundVerts) / 3)

    glVertexPointer(3, GL_FLOAT, 0, groundVerts)
    glColorPointer(3, GL_FLOAT, 0, groundColors)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glDrawArrays(GL_QUADS, 0, numVerts)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)

    # cloth
    if not paused and frame > 200: # FIXME
        cloth.simulate()

    cloth.updateMesh()
    cloth.render()


glsl.onInit(init)
glsl.onRender(render)


def get_mouse_ray(x: int, y: int) -> Tuple[wp.vec3, wp.vec3]:
    viewport = glGetIntegerv(GL_VIEWPORT)
    model_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
    proj_matrix = glGetDoublev(GL_PROJECTION_MATRIX)

    y = viewport[3] - y - 1
    p0 = gluUnProject(x, y, 0.0, model_matrix, proj_matrix, viewport)
    p1 = gluUnProject(x, y, 1.0, model_matrix, proj_matrix, viewport)
    origin = wp.vec3(p0[0], p0[1], p0[2])
    direction = wp.sub(wp.vec3(p1[0], p1[1], p1[2]), origin)
    direction = wp.normalize(direction)
    return origin, direction


class Mouse(shaderbang.input.Mouse):
    mouseX, mouseY = 0, 0
    click = False
    particle: Optional[Particle] = None

    def init(self, width, height, **kwargs):
        super().init(width=width, height=height, **kwargs)
        self.mouseX = width // 2
        self.mouseY = height // 2

    def event(self, ev, **__):
        if ev.matches(EV_KEY):
            if not ev.code == EV_KEY.BTN_LEFT:
                return
            if ev.value == 1:
                self.click = True
                self.drag = True
            elif ev.value == 0:
                if self.particle:
                    cloth.end_drag(self.particle)
                self.drag = False
                self.particle = None
        elif ev.matches(EV_REL):
            if ev.code == EV_REL.REL_X:
                self.mouseX = max(1, min(self.mouseX + ev.value, self.resolution[0]))
            elif ev.code == EV_REL.REL_Y:
                self.mouseY = max(1, min(self.mouseY + ev.value, self.resolution[1]))

    def render(self, **_):
        if self.click:
            (origin, direction) = get_mouse_ray(self.mouseX, self.mouseY)
            self.particle = cloth.start_drag(origin, direction)
            self.click = False
        elif self.drag:
            if self.particle:
                (origin, direction) = get_mouse_ray(self.mouseX, self.mouseY)
                cloth.drag(origin, direction, self.particle)


class ParticleSlot(TouchSlot):
    particle: Optional[Particle] = None

    @TouchSlot.drag.setter
    def drag(self, value):
        if self._drag and not value and self.particle:
            cloth.end_drag(self.particle)
        self._drag = value


class Touchscreen(shaderbang.input.Touchscreen[ParticleSlot]):

    def __init__(self, name: str, dev: Device):
        super().__init__(name, dev, ParticleSlot)

    def render(self, **_):
        for slot in self.slots:
            if slot.touch:
                (origin, direction) = get_mouse_ray(slot.touchX, slot.touchY)
                slot.particle = cloth.start_drag(origin, direction)
                slot.touch = False
            elif slot.drag:
                if slot.particle:
                    (origin, direction) = get_mouse_ray(slot.touchX, slot.touchY)
                    cloth.drag(origin, direction, slot.particle)


parser = argparse.ArgumentParser(description='Run cloth simulation')
parser.add_argument('--async-page-flip', action=argparse.BooleanOptionalAction,
                    help='use async page flipping')
parser.add_argument('--atomic-drm-mode', action=argparse.BooleanOptionalAction,
                    help='use atomic mode setting and fencing')
parser.add_argument('-C', '--connector', metavar='CONNECTOR', type=int,
                    help='the DRM connector')
parser.add_argument('-D', '--device', metavar='DEVICE', type=Path,
                    help='the DRM device')
parser.add_argument('--mode', metavar='MODE', type=str,
                    help='specify the video mode in the format <resolution>[-<vrefresh>]')
parser.add_argument('-n', '--frames', metavar='N', type=int,
                    help='run for the given number of frames and exit')
args = parser.parse_args()


def input_from_device(dev: Device):
    if dev.has(EV_REL) and dev.has(EV_KEY.BTN_LEFT):
        # Mouse
        Mouse(dev.name, dev)
        pass
    elif dev.has(EV_KEY) and dev.has(EV_KEY.KEY_A):
        # Keyboard
        # Keyboard(args.keyboard if args.keyboard else 'iKeyboard', dev)
        pass
    elif dev.has(EV_ABS.ABS_MT_SLOT) and dev.has(EV_KEY.BTN_TOUCH) and dev.has_property(INPUT_PROP_DIRECT):
        # Touchscreen
        # Only consider direct input devices, like touchscreens and drawing tablets, see:
        # https://www.kernel.org/doc/Documentation/input/event-codes.txt
        Touchscreen(dev.name, dev)
    elif dev.has(EV_ABS.ABS_MT_SLOT) and dev.has(EV_KEY.BTN_TOUCH) and dev.has_property(INPUT_PROP_POINTER):
        # Trackpad
        # https://www.kernel.org/doc/Documentation/input/multi-touch-protocol.txt
        # Trackpad(args.trackpad if args.trackpad else 'iTrackpad', dev)
        pass
    else:
        dev.fd.close()


devices = ExitStack()
with devices:
    for path in list(filter(lambda p: os.path.exists(p) and stat.S_ISCHR(os.stat(p)[stat.ST_MODE]),
                            glob.glob('{}/event*'.format('/dev/input')))):
        input_from_device(Device(devices.enter_context(open(path, 'rb'))))
    devices = devices.pop_all()

inotify = INotify()
inotify.add_watch('/dev/input', IN_CREATE | IN_ATTRIB)


def hot_plug_devices():
    with devices:
        while True:
            for ev in inotify.read():
                p = os.path.join('/dev/input', ev.name)
                if (str.startswith(ev.name, 'event')
                        and os.path.exists(p)
                        and os.access(p, os.R_OK)
                        and stat.S_ISCHR(os.stat(p)[stat.ST_MODE])):
                    input_from_device(Device(devices.enter_context(open(p, 'rb'))))


Thread(target=hot_plug_devices, daemon=True).start()

ret = glsl.init(bytes("", 'utf-8'), byref(options(args)))
if ret != 0:
    devices.close()
    exit(ret)

ret = glsl.run()
if ret != 0:
    devices.close()
    exit(ret)

stopped = threading.Event()
pthread_sigmask(signal.SIG_BLOCK, [signal.SIGCONT])


def join():
    glsl.join()
    stopped.set()
    pthread_kill(main_thread().ident, signal.SIGCONT)


Thread(target=join, daemon=True).start()

if sigwait({signal.SIGINT, signal.SIGCONT}) == signal.SIGINT:
    glsl.stop()
    ret = stopped.wait(timeout=5.0)

inotify.close()
devices.close()
