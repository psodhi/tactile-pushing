# Install headers
file(GLOB contact_headers "*.h")
install(FILES ${contact_headers} DESTINATION include/pushestcpp/contact)

# Build tests
gtsamAddTestsGlob(contact "tests/*.cpp" "" ${PROJECT_NAME})