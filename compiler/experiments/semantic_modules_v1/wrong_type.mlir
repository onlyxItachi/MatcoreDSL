module {
  func.func private @library(f32, f32) -> f32
  func.func @caller(%a: f32, %b: f32) -> i32 {
    %r = call @library(%a, %b) : (f32, f32) -> i32
    return %r : i32
  }
}
