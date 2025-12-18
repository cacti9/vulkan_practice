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

// Author: @patriciogv
// Title: 4 cells DF

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    //st.y = 1.0 - st.y; // flip vertical coord
    float3 color = float3(0.0);

    // Cell positions
    float2 point[5];
    point[0] = float2(0.83,0.75);
    point[1] = float2(0.60,0.07);
    point[2] = float2(0.28,0.64);
    point[3] =  float2(0.31,0.26);
    point[4] = ubo.mousePos/ubo.resolution;

    float m_dist = 1.;  // minimum distance

    // Iterate through the points positions
    for (int i = 0; i < 5; i++) {
        float dist = distance(st, point[i]);

        // Keep the closer distance
        m_dist = min(m_dist, dist);
    }

    // Draw the min distance (distance field)
    color += m_dist;

    // Show isolines
    // color -= step(.7,abs(sin(50.0*m_dist)))*.3;

    return float4(color,1.0);
}