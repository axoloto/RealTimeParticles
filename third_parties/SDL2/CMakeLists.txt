find_library(SDL SDL2 PATHS "$ENV{TP_DIR}/SDL2/lib" REQUIRED)
find_library(SDLmain SDL2main PATHS "$ENV{TP_DIR}/SDL2/lib" REQUIRED)

if(NOT SDL)
  message(FATAL_ERROR "SDL library not found")
endif()
if(NOT SDLmain)
  message(FATAL_ERROR "SDLmain library not found")
endif()

include_directories("include")
