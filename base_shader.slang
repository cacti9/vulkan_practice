struct VSInput {
    float2 inPosition;
    float3 inColor;
};

struct UniformBuffer {
    float2 resolution;
    float2 mousePos;
    float  time;
};
ConstantBuffer<UniformBuffer> ubo;

struct VSOutput
{
    float4 pos : SV_Position;
    float3 color;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pos = float4(input.inPosition, 0.0, 1.0);
    output.color = input.inColor;
    return output;
}

float mod(float x, float y) {
    return x - y*floor(x/y);
}

// 2D Random
float random(in int2 st)
{
    // Cast to unsigned
    uint x = (uint)st.x;
    uint y = (uint)st.y;

    // Mix (PCG-ish / avalanche style)
    uint h = x * 0x8da6b343u ^ y * 0xd8163841u;
    h ^= (h >> 16);
    h *= 0x7feb352du;
    h ^= (h >> 15);
    h *= 0x846ca68bu;
    h ^= (h >> 16);

    // Map to [0,1) using 24 high bits (safe in float)
    return (h >> 8) * (1.0 / 16777216.0); // 2^24
}
// 2D Noise based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
float noise (in float2 st) {
    int2 i = (int2)floor(st);
    float2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + int2(1, 0));
    float c = random(i + int2(0, 1));
    float d = random(i + int2(1, 1));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    float2 u = f*f*(3.0-2.0*f);
    // u = smoothstep(0.,1.,f);

    // Mix 4 coorners percentages
    return lerp(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord
    float3 color = float3(0.0);

    // Scale the coordinate system to see
    // some noise in action
    float2 pos = float2(st*50.0);

    // Use the noise function
    color = float3(noise(pos));

    return float4(color,1.0);
}