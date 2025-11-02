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

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord

    float3 colorA = float3(0.149,0.141,0.912);
    float3 colorB = float3(1.000,0.833,0.224);

    const float gradient=2.0;
    float mountain = sin(st.x * PI * 1.7 + 0.4)*0.04 + 0.08;
    float mountainColor = clamp((st.y - mountain)*gradient, 0.0, 1.0);
    float sky = sqrt(st.x+0.3)*0.3+0.75;
    float skyColor = clamp((sky - st.y)*gradient, 0.0, 1.0);
    float mixed = min(mountainColor,skyColor);
    float3 color = lerp(colorA,colorB,mixed/(1.0+pow(clamp(ubo.time-7.0,0.0,1.0),0.5)*0.3+ubo.time*0.02));
    float2 sunPosition = float2(0.80746, 0.05*ubo.time-0.3);
    float2 diff = sunPosition - st;
    float pct = clamp(sqrt(mul(diff,diff))*
        clamp(pow(abs(3.0-ubo.time), 1.5),1.0, 8.0)
        ,0.0,1.0);
    pct = lerp(pct, 1.0, float(st.y<mountain));
    color = lerp( float3(0.9,0.05,0.02), color,pct);
    return float4(color, 1.0);
}