set(pushestcpp_subdirs 
    contact
    dynamics
    geometry
    thirdparty
)
set(pushestcpp_srcs)

# files want to be excluded
set(excluded_sources "")

# Library sources
foreach(subdir ${pushestcpp_subdirs})
  file(GLOB subdir_srcs "${subdir}/*.cpp" "${subdir}/*.h")
  list(REMOVE_ITEM subdir_srcs "${excluded_sources}")

  file(GLOB subdir_test_files "${subdir}/tests/*")
  
  list(APPEND pushestcpp_srcs ${subdir_srcs})
  message(STATUS "Building Module: ${subdir}")

  # local and tests
  add_subdirectory(${subdir})
endforeach(subdir)

# build static/shared lib
if(PUSHESTCPP_BUILD_STATIC_LIBRARY)
  # static
  message(STATUS "Build static library")
  add_library(${PROJECT_NAME} STATIC ${pushestcpp_srcs})
  target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${GTSAM_LIBRARIES} ${TORCH_LIBRARIES})
  set_target_properties(pushestcpp PROPERTIES
    OUTPUT_NAME         pushestcpp
    LINKER_LANGUAGE CXX
    CLEAN_DIRECT_OUTPUT 1)
    install(TARGETS ${PROJECT_NAME} EXPORT pushestcpp-exports ARCHIVE DESTINATION lib)
else()
  # shared
  message(STATUS "Build shared library")
  add_library(${PROJECT_NAME} SHARED ${pushestcpp_srcs})
  target_link_libraries(pushestcpp ${Boost_LIBRARIES} ${GTSAM_LIBRARIES} ${TORCH_LIBRARIES})
  set_target_properties(pushestcpp PROPERTIES
    OUTPUT_NAME         ${PROJECT_NAME}
    LINKER_LANGUAGE CXX
    CLEAN_DIRECT_OUTPUT 1)
    install(TARGETS ${PROJECT_NAME} EXPORT pushestcpp-exports LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
endif()

list(APPEND PUSHESTCPP_EXPORTED_TARGETS pushestcpp)
set(PUSHESTCPP_EXPORTED_TARGETS "${PUSHESTCPP_EXPORTED_TARGETS}" PARENT_SCOPE)