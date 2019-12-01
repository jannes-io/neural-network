SRC_DIR=./src
OBJ_DIR=./.obj
BUILD_DIR=./build

NAME=$(BUILD_DIR)/neural-net
all: $(NAME)

CC=gcc
CFLAGS=-Wall -Wextra -pedantic

CXX=g++
CXXFLAGS=-Wall -Wextra -pedantic --std=c++17

LINK=-lstdc++

STNAMES=\
		main \
		Net \
		Neuron \
		TrainingData \

OBJECTS=$(patsubst %,$(OBJ_DIR)/%.o, $(STNAMES))

ifeq ($(OS), Windows_NT)
define mkdir =
  	if not exist "$(1)" mkdir "$(1)"
endef
else
define mkdir
	mkdir -p $(1)
endef
endif

$(NAME): $(OBJECTS)
	@$(call mkdir,$(BUILD_DIR))
	$(CC) -o $@ $(OBJECTS) $(LINK)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@$(call mkdir,$(OBJ_DIR))
	$(CC) -o $@ -c $< $(CFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@$(call mkdir,$(OBJ_DIR))
	$(CXX) -o $@ -c $< $(CXXFLAGS)
