add_executable(matcore-extract main.cpp)
target_compile_features(matcore-extract PRIVATE cxx_std_20)
target_link_libraries(matcore-extract PRIVATE matcore_frontend_bootstrap)
