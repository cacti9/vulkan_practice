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

// Author @patriciogv ( patriciogonzalezvivo.com ) - 2015
float2 brickTile(float2 _st, float _zoom){
    _st *= _zoom;
    float time = ubo.time/1.0;
    float move = step(1., mod(time,2.0));

    // Here is where the offset is happening
    _st.x += step(1., mod(_st.y,2.0)) * fract(time) * move;
    _st.y += step(1., mod(_st.x,2.0)) * fract(time) * (1.0-move);

    return fract(_st);
}

float box(float2 _st, float2 _size){
    _size = float2(0.5)-_size*0.5;
    float2 uv = smoothstep(_size,_size+float2(1e-4),_st);
    uv *= smoothstep(_size,_size+float2(1e-4),float2(1.0)-_st);
    return uv.x*uv.y;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord
    float3 color = float3(0.0);

    // Apply the brick tiling
    st = brickTile(st,5.0);

    color = float3(box(st,float2(0.9)));

    // Uncomment to see the space coordinates
    // color = float3(st,0.0);

    return float4(color,1.0);
}