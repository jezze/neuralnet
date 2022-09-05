.PHONY: all clean

BIN:=nn
OBJ:=main.o
LIBS:=-lm
CFLAGS:=-Wall -Werror
PREFIX:=/usr/local/bin

all: $(BIN)

.c.o:
	$(CC) -c -o $@ $(CFLAGS) $<

$(BIN): $(OBJ)
	$(CC) -o $@ $^ $(LIBS)

clean:
	rm -rf $(BIN) $(OBJ)

install: $(BIN)
	install -m 755 $(BIN) $(PREFIX)

