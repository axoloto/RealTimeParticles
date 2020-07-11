message(STATUS "version 0.1")

find_library(SDL SDL2 PATHS "$ENV{TP_DIR}/SDL2/lib" REQUIRED)
find_library(SDLmain SDL2main PATHS "$ENV{TP_DIR}/SDL2/lib" REQUIRED)

set(SDL2_DLL "$ENV{TP_DIR}/SDL2/lib/SDL2.dll")

#set_target_properties(SDL2 PROPERTIES LINKER_LANGUAGE C)

if(NOT SDL)
  message(FATAL_ERROR "SDL library not found")
endif()
if(NOT SDLmain)
  message(FATAL_ERROR "SDLmain library not found")
endif()


include_directories("include")
message(STATUS "version 0.1")
