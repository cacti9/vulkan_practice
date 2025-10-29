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

static const float PI = 3.1415926535;

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord
    const float B = 2.0;
    st.y = st.y*2.0*B - B; // [0,1] -> [-B,B]
    const float A = 2.0*PI;
    st.x = st.x*2.0*A - A; // [0,1] -> [-A, A]
    // y = f(2Ax - A)/2B + 0.5; y' = f'(2Ax - A)*A/B

    float y=sin(st.x);
    float derivative = A/B*cos(st.x);
    float dist = abs(st.y-y) / sqrt(derivative*derivative + 1.0);
    float thickness = 0.004;

    float color = 1.0 - step(thickness, dist);

    return float4(float3(color), 1.0);
}