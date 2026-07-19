cmake_minimum_required(VERSION 3.24)

project(MatcoreDSLInstalledConsumer LANGUAGES CXX)

find_package(MatcoreDSL 0.1 CONFIG REQUIRED)

matcoredsl_add_executable(
  matcore_consumer
  SOURCE consumer.mdsl
  CXX_STANDARD 20
  MATCORE_TARGET cpu
  FRONTEND native
)

enable_testing()
add_test(NAME matcore_consumer.runs COMMAND matcore_consumer)
