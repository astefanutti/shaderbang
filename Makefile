CC=gcc
CFLAGS=-c -g -Wall -O3 -Winvalid-pch -Wextra -std=gnu99 -fPIC -fdiagnostics-color=always -pipe -pthread -I/usr/include/libdrm
LDFLAGS=-Wl,--no-as-needed -lGLESv2 -Wl,--as-needed,--no-undefined
LDLIBS=-lGLESv2 -lEGL -ldrm -lgbm -lxcb-randr -lxcb -lpthread
SOURCES=common.c drm-atomic.c drm-common.c drm-legacy.c glsl.c lease.c perfcntrs.c shaderbang.c shadertoy.c
OBJECTS=$(SOURCES:%.c=%.o)
EXECUTABLE=glsl
LIBRARY=glsl.so

all: $(SOURCES) $(EXECUTABLE) $(LIBRARY)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

$(LIBRARY): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -shared -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean :
	rm -f *.o $(EXECUTABLE) $(LIBRARY)
