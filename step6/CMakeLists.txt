cmake_minimum_required(VERSION 3.10)
set(TARGET "step6")
set(CMAKE_CXX_STANDARD 20)
find_package(deal.II CONFIG REQUIRED)

set(TARGET_SRC ${TARGET}.cpp)

deal_ii_initialize_cached_variables()
project(${TARGET} LANGUAGES CXX)
deal_ii_invoke_autopilot()
