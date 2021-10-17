# Install headers
file(GLOB utils_headers "*.h")
install(FILES ${utils_headers} DESTINATION include/pushest/thirdparty)