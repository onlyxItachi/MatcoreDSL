// Research-only upstream function mechanics, not a Matcore import format.
module {
  func.func private @library(%a: f32, %b: f32) -> f32 {
    %r = arith.addf %a, %b : f32
    return %r : f32
  }
  func.func @caller(%a: f32, %b: f32) -> f32 {
    %r = call @library(%a, %b) : (f32, f32) -> f32
    return %r : f32
  }
}
