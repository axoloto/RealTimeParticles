macro(run_conan)
  # Download automatically, you can also just copy the conan.cmake file
  if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake"
                TLS_VERIFY ON)
  endif()

  set(ENV{CONAN_REVISIONS_ENABLED} 1)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
  list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

  include(${CMAKE_BINARY_DIR}/conan.cmake)

  # Latest version before Conan 2.0
  conan_check(VERSION 1.59.0 REQUIRED)

  conan_add_remote(NAME cci URL https://center.conan.io INDEX 0)
  conan_add_remote(NAME bincrafters URL https://bincrafters.jfrog.io/artifactory/api/conan/public-conan)

  # For multi configuration generators, like VS and XCode
  if(NOT CMAKE_CONFIGURATION_TYPES OR NOT MULTI_CONAN_BUILD)
    message(STATUS "Single configuration build!")
    set(LIST_OF_BUILD_TYPES ${CMAKE_BUILD_TYPE})
  else()
    message(STATUS "Multi-configuration build: '${CMAKE_CONFIGURATION_TYPES}'!")
    set(LIST_OF_BUILD_TYPES ${CMAKE_CONFIGURATION_TYPES})
  endif()

  foreach(TYPE ${LIST_OF_BUILD_TYPES})
    message(STATUS "Running Conan for build type '${TYPE}'")

    conan_cmake_autodetect(settings BUILD_TYPE ${TYPE})

    conan_cmake_install(PATH_OR_REFERENCE ${CMAKE_SOURCE_DIR}
                        BUILD missing
                        SETTINGS ${settings})
  endforeach()

endmacro()
