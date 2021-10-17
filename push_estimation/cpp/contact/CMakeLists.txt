# Install headers
file(GLOB contact_headers "*.h")
install(FILES ${contact_headers} DESTINATION include/cpp/contact)