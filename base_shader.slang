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
#define PI 3.14159265358979323846

float2 rotate2D(float2 _st, float _angle){
    _st -= 0.5;
    _st =  mul(float2x2(cos(_angle),-sin(_angle),
                sin(_angle),cos(_angle)), _st);
    _st += 0.5;
    return _st;
}

float2 tile(float2 _st, float _zoom){
    _st *= _zoom;
    return fract(_st);
}

float box(float2 _st, float2 _size, float _smoothEdges){
    _size = float2(0.5)-_size*0.5;
    float2 aa = float2(_smoothEdges*0.5);
    float2 uv = smoothstep(_size,_size+aa,_st);
    uv *= smoothstep(_size,_size+aa,float2(1.0)-_st);
    return uv.x*uv.y;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord
    float3 color = float3(0.0);

    // Divide the space in 4
    st = tile(st,4.);

    // Use a matrix to rotate the space 45 degrees
    st = rotate2D(st,PI*0.25);

    // Draw a square
    color = float3(box(st,float2(0.7),0.01));
    // color = float3(st,0.0);

    //my modification
    st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord
    st = tile(st, 2.);
    color *= float3(st,1.0);

    return float4(color,1.0);
}