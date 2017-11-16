set(MODULE src)

set(SRCS
  CA/Cell.cxx
  CA/Cluster.cxx
  CA/Event.cxx
  CA/IOUtils.cxx
  CA/Label.cxx
  CA/Layer.cxx
  CA/PrimaryVertexContext.cxx
  CA/Road.cxx
  CA/Tracker.cxx
  CA/TrackingUtils.cxx
  CA/Tracklet.cxx
)

include_directories(${TRACKING-ITSU_SOURCE_DIR}/include)


if(TRACKINGITSU_TARGET_DEVICE STREQUAL OPEN_CL)
  LIST(APPEND SRCS
    CA/opencl/Context.cpp
    CA/opencl/PrimaryVertexContext.cpp
    CA/opencl/Stream.cpp
    CA/opencl/Utils.cpp
    CA/opencl/Tracker.cpp
  )
configure_file(CA/opencl/kernel/fillIndexTables.cl ${CMAKE_CURRENT_BINARY_DIR}/kernel/fillIndexTables.cl COPYONLY)
  include_directories( ${OPENCL_INCLUDE_DIRS} ${TRACKING-ITSU_SOURCE_DIR}/src/CA/opencl/kernel)

endif(TRACKINGITSU_TARGET_DEVICE STREQUAL OPEN_CL)

	add_library(${MODULE} ${SRCS})
endif(TRACKINGITSU_TARGET_DEVICE STREQUAL GPU_CUDA)
