set(TARGET llama-tensor-debug)
add_executable(${TARGET} tensor-debug.cpp)
target_include_directories(${TARGET} PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
