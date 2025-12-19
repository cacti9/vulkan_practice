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
// Title: CellularNoise
float2 random2( float2 p ) {
    return fract(sin(float2(dot(p,float2(127.1,311.7)),dot(p,float2(269.5,183.3))))*43758.5453);
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    //st.y = 1.0 - st.y; // flip vertical coord
    float3 color = float3(0.0);

    // Scale
    st *= 50.;

    // Tile the space
    float2 i_st = floor(st);
    float2 f_st = fract(st);

    float m_dist = 1.;  // minimum distance
    float2 m_point;

    for (int y= -1; y <= 1; y++) {
        for (int x= -1; x <= 1; x++) {
            // Neighbor place in the grid
            float2 neighbor = float2(float(x),float(y));

            // Random position from current + neighbor place in the grid
            float2 point = random2(i_st + neighbor);

			// Animate the point
            point = 0.5 + 0.5*sin(ubo.time + 6.2831*point);

			// Vector between the pixel and the point
            float2 diff = neighbor + point - f_st;

            // Distance to the point
            float dist = length(diff);

            // Keep the closer distance
            //m_dist = min(m_dist, dist);
            if (m_dist > dist) {
                m_dist = dist;
                m_point = point;
            }
        }
    }

    // Draw the min distance (distance field)
    color += m_dist;
    color.rg = m_point;

    // Draw cell center
    color += 1.-step(.02, m_dist);

    // Draw grid
    color.r += step(.98, f_st.x) + step(.98, f_st.y);

    // Show isolines
    // color -= step(.7,abs(sin(27.0*m_dist)))*.5;

    return float4(color,1.0);
}