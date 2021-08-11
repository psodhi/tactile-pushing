# Install headers
file(GLOB geometry_headers "*.h")
install(FILES ${geometry_headers} DESTINATION include/pushestcpp/geometry)

# Build tests
gtsamAddTestsGlob(geometry "tests/*.cpp" "" ${PROJECT_NAME})