CC=gcc
CFLAGS=-c -g -Wall -O3 -Winvalid-pch -Wextra -std=gnu99 -fPIC -fdiagnostics-color=always -pipe -pthread -I/usr/include/libdrm
LDFLAGS=-Wl,--no-as-needed -lGLESv2 -Wl,--as-needed,--no-undefined
LDLIBS=-lGLESv2 -lEGL -ldrm -lgbm -lpthread #-lxcb-randr -lxcb

SRC_DIR=shaderbang/lib
SOURCES=common.c drm-atomic.c drm-common.c drm-legacy.c lease.c shaderbang.c
OBJECTS=$(addprefix $(SRC_DIR)/,$(SOURCES:.c=.o))
LIBRARY=shaderbang/_shaderbang.so

all: $(LIBRARY)

$(LIBRARY): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -shared -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $< -o $@

clean :
	rm -f $(SRC_DIR)/*.o $(LIBRARY)
