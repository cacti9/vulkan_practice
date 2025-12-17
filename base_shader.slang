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

// Author @patriciogv ( patriciogonzalezvivo.com ) - 2015
float box(in float2 _st, in float2 _size){
    _size = float2(0.5) - _size*0.5;
    float2 uv = smoothstep(_size,
                        _size+float2(0.001),
                        _st);
    uv *= smoothstep(_size,
                    _size+float2(0.001),
                    float2(1.0)-_st);
    return uv.x*uv.y;
}

float cross(in float2 _st, float _size){
    return  box(_st, float2(_size,_size/4.)) +
            box(_st, float2(_size/4.,_size));
}

#define PI 3.14159265359
float2x2 rotate2d(float _angle){
    return float2x2(cos(_angle),-sin(_angle),
                sin(_angle),cos(_angle));
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord
    float3 color = float3(0.0);

    st -= float2(0.5);
    st = mul(rotate2d(sin(ubo.time)*PI),st);
    st += float2(0.5);

    // Show the coordinates of the space on the background
    color = float3(st.x,st.y,0.0);

    // Add the shape on the foreground
    color += float3(cross(st,0.4));

    return float4(color,1.0);
}