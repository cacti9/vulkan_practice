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

#define PI 3.14159265359
#define TWO_PI 6.28318530718

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord
    float3 color = float3(0.0);
    float d = 0.0;

    // Remap the space to -1. to 1.
    st = st *2.-1.;

    // Number of sides of your shape
    int N = 8;

    // Angle and radius from the current pixel
    float a = atan2(st.x,st.y)+PI;
    float r = TWO_PI/float(N);

    // Shaping function that modulate the distance
    d = cos(floor(.5+a/r)*r-a)*length(st);

    color = float3(1.0-smoothstep(.4,.41,d));

    return float4(color,1.0);
}