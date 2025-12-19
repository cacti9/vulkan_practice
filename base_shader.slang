struct VSInput {
    float2 inPosition;
    float3 inColor;
};

struct UniformBuffer {
    float2 resolution;
    float2 mousePos;
    float2 bottomLeft;
    float2 span;
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

// square of complex number z=x+yi
float2 square(float2 z) {
    return float2(
        z.x*z.x - z.y*z.y,
        2*z.x*z.y
    );
}
float3 palette(float t) {
    return 0.5 + 0.5*cos(6.28318*(float3(0.0, 0.33, 0.67) + t));
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord
    
    float3 color = float3(0.0);
    float2 z=float2(0.);
    float2 c = ubo.bottomLeft + ubo.span*st;
    int IT_COUNT=50;
    for (int i=0; i<IT_COUNT; ++i) {
        if (length(z) >= 2.) {
            float r2 = dot(z,z);
            float v = i+1 - log(log(r2));
            color = palette(v*0.03+.6);
            break;
        }
        z = square(z) + c;
    }

    return float4(color,1.0);
}