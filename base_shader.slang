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
static const float thickness = 0.003;
static const float edge = 0.001;

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord
    const float B = 2.0;
    st.y = st.y*2.0*B - B; // [0,1] -> [-B,B]
    const float A = 2.0*PI;
    st.x = st.x*2.0*A - A; // [0,1] -> [-A, A]
    // y = f(2Ax - A)/2B + 0.5; y' = f'(2Ax - A)*A/B
    float x_axis = smoothstep(thickness*B + edge, 0.0, abs(st.y));
    float y_axis = smoothstep(thickness*A + edge, 0.0, abs(st.x));
    float axes = max(x_axis, y_axis);
    float3 color = float3(float2(1.0 - axes), 1.0);


    float f1=sin(st.x*PI+ubo.time);
    float df1 = A/B*cos(st.x*PI+ubo.time)*PI;
    float dist = abs(st.y-f1) / sqrt(df1*df1 + 1.0);

    float f1color = 1.0 - smoothstep(thickness+edge, thickness-edge, dist);
    color = lerp(float3(f1color), color, f1color);

    return float4(float3(color), 1.0);
}