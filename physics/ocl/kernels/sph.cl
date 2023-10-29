
// Most defines are in define.cl
// define.cl must be included as first file.cl to create OpenCL program

/*
  Poly6 kernel introduced in
  Muller et al. 2003. "Particle-based fluid simulation for interactive applications"
  Return null value if vec length is superior to effectRadius
*/
inline float poly6(const float4 vec, const float effectRadius)
{
  float vecLength = fast_length(vec);
  return (1.0f - step(effectRadius, vecLength)) * POLY6_COEFF * pow((effectRadius * effectRadius - vecLength * vecLength), 3);
}

inline float poly6L(const float vecLength, const float effectRadius)
{
  return (1.0f - step(effectRadius, vecLength)) * POLY6_COEFF * pow((effectRadius * effectRadius - vecLength * vecLength), 3);
}

/*
  Jacobian (on vec coords) of Spiky kernel introduced in
  Muller et al. 2003. "Particle-based fluid simulation for interactive applications"
  Return null vector if vec length is superior to effectRadius
*/
inline float4 gradSpiky(const float4 vec, const float effectRadius)
{
  const float vecLength = fast_length(vec);

  if(vecLength <= FLOAT_EPS)
    return (float4)(0.0f);

  return vec * (1.0f - step(effectRadius, vecLength)) * SPIKY_COEFF * -3 * pow((effectRadius - vecLength), 2) / vecLength;
}
