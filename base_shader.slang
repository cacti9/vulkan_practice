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

// Author @patriciogv - 2015 - patriciogonzalezvivo.com
float2 skew (float2 st) {
    float2 r = float2(0.0);
    r.x = 1.1547*st.x;
    r.y = st.y+0.5*r.x;
    return r;
}

float3 simplexGrid (float2 st) {
    float3 xyz = float3(0.0);

    float2 p = fract(skew(st));
    if (p.x > p.y) {
        xyz.xy = 1.0-float2(p.x,p.y-p.x);
        xyz.z = p.y;
    } else {
        xyz.yz = 1.0-float2(p.x-p.y,p.y);
        xyz.x = p.x;
    }

    return fract(xyz);
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord
    float3 color = float3(0.0);

    // Scale the space to see the grid
    st *= 100.;

    // Show the 2D grid
    color.rg = fract(st);

    // Skew the 2D grid
     color.rg = fract(skew(st));

    // Subdivide the grid into to equilateral triangles
     color = simplexGrid(st);

    return float4(color,1.0);
}