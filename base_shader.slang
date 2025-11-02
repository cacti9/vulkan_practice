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
#define TWO_PI 6.28318530718

float mod(float x, float y) {
    return x - y*floor(x/y);
}
float3 mod(float3 x, float y) {
    return x - y*floor(x/y);
}
//  Function from Iñigo Quiles
//  https://www.shadertoy.com/view/MsS3Wc
float3 hsb2rgb( in float3 c ){
    float3 rgb = clamp(abs(mod(c.x*6.0+float3(0.0,4.0,2.0),
                             6.0)-3.0)-1.0,
                     0.0,
                     1.0 );
    rgb = rgb*rgb*(3.0-2.0*rgb);
    return c.z * lerp( float3(1.0), rgb, c.y);
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float2 st = vertIn.pos.xy/ubo.resolution;
    st.y = 1.0 - st.y; // flip vertical coord
    float3 color = float3(0.0);

    // Use polar coordinates instead of cartesian
    float2 toCenter = float2(0.5)-st;
    float angle = atan2(toCenter.y,toCenter.x)+ubo.time+PI;
    angle = mod(angle, TWO_PI);
    float radius = length(toCenter)*2.0;

    // Map the angle (0 to 2PI) to the Hue (from 0 to 1)
    // and the Saturation to the radius
    float a=0.9;
    float b=1.0;
    color = hsb2rgb(float3(smoothstep(0.0,1.0,(angle/TWO_PI)),radius,1.0));

    return float4(color, 1.0);
}