struct VSInput {
    float2 inPosition;
    float3 inColor;
};

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

#define L 8          // number of 32-bit limbs (example: 256-bit)
#define FRAC_LIMBS 7 // N = 32*7= 224 fractional bits

struct BigUINT {
    uint limb[L]; // little-endian
};
bool isZero(BigUINT a) {
    for (int i = 0; i < L; i++)
        if (a.limb[i] != 0) return false;
    return true;
}
BigUINT operator+(BigUINT a, BigUINT b) {
    BigUINT r;
    uint carry = 0;
    for(int i=0;i<L;i++)
    {
        uint64_t s = (uint64_t)a.limb[i] + (uint64_t)b.limb[i] + (uint64_t)carry;
        r.limb[i] = (uint)s;
        carry = (uint)(s >> 32);
    }
    return r;
}
// must be a>=b
BigUINT operator-(BigUINT a, BigUINT b) {
    BigUINT r;
    uint borrow=0;

    for(int i=0;i<L;i++)
    {
        uint ai = a.limb[i];
        uint bi = b.limb[i];

        // subtract bi + borrow from ai
        uint t = bi + borrow;

        // detect borrow: if ai < t, we need to borrow from next limb
        uint newBorrow = (ai < t) ? 1u : 0u;

        r.limb[i] = ai - t;
        borrow = newBorrow;
    }
    return r;
}
BigUINT shiftLeft(BigUINT a) {
    BigUINT r;
    uint carry=0;

    for(int i=0;i<L;i++)
    {
        uint newCarry = (a.limb[i] >> 31) & 1u;
        r.limb[i] = (a.limb[i]<<1) | carry;
        carry = newCarry;
    }

    return r;
}
BigUINT operator>>(BigUINT a, int N) {
    BigUINT r;

    uint wordShift = N >> 5;   // /32
    uint bitShift  = N & 31u;  // %32

    // default to zero
    for (int i = 0; i < L; ++i) r.limb[i] = 0u;

    // If shifting out everything -> zero
    if (wordShift >= uint(L)) return r;

    // For each destination limb i, read from source limb (i + wordShift)
    // because limb[0] is least significant.
    for (int i = 0; i < L; ++i)
    {
        int src = i + int(wordShift);
        if (src >= L) break;

        uint lo = a.limb[src];

        if (bitShift == 0u)
        {
            r.limb[i] = lo;
        }
        else
        {
            // bring in high bits from next limb (more significant limb)
            uint hi = (src + 1 < L) ? a.limb[src + 1] : 0u;
            r.limb[i] = (lo >> bitShift) | (hi << (32u - bitShift));
        }
    }

    return r;
}
int compare(BigUINT a, BigUINT b) {
    for(int i=L-1; i>=0; i--)
    {
        uint ai = a.limb[i];
        uint bi = b.limb[i];
        if (ai > bi) return 1;
        if (ai < bi) return -1;
    }
    return 0;
}

struct BigUINT2 {
    uint limb[2*L];
};
BigUINT2 mulWide(BigUINT a, BigUINT b) {
    BigUINT2 p;
    for(int i=0;i<2*L;i++) p.limb[i]=0;

    for(int i=0;i<L;i++)
    {
        uint64_t carry = 0;
        for(int j=0;j<L;j++)
        {
            int k = i + j;
            uint64_t cur = (uint64_t)p.limb[k]
                         + (uint64_t)a.limb[i] * (uint64_t)b.limb[j]
                         + carry;
            p.limb[k] = (uint)cur;
            carry = cur >> 32;
        }

        // propagate carry across higher limbs
        int k = i + L;
        while((bool)carry && k < 2*L)
        {
            uint64_t cur2 = (uint64_t)p.limb[k] + carry;
            p.limb[k] = (uint)cur2;
            carry = cur2 >> 32;
            k++;
        }
    }
    return p;
}
BigUINT2 squareWide(BigUINT a) {
    BigUINT2 p;
    for(int i=0;i<2*L;i++) p.limb[i]=0;

    for(int i=0;i<L;i++)
    {
        uint64_t carry = 0;
        for(int j=0;j<L;j++)
        {
            int k = i + j;
            uint64_t cur = (uint64_t)p.limb[k]
                         + (uint64_t)a.limb[i] * (uint64_t)a.limb[j]
                         + carry;
            p.limb[k] = (uint)cur;
            carry = cur >> 32;
        }

        // propagate carry across higher limbs
        int k = i + L;
        while((bool)carry && k < 2*L)
        {
            uint64_t cur2 = (uint64_t)p.limb[k] + carry;
            p.limb[k] = (uint)cur2;
            carry = cur2 >> 32;
            k++;
        }
    }
    return p;
}

struct Fix {
    BigUINT mag;
    bool neg=false;

};
Fix negate(Fix x) {
    x.neg = !x.neg;
    return x;
}
Fix normalizeFix(Fix x) {
    if (isZero(x.mag)) x.neg = false;
    return x;
}
Fix operator+(Fix x, Fix y) {
    Fix r;
    if (x.neg==y.neg)
    {
        r.neg=x.neg;
        r.mag=x.mag+y.mag;
    } else {
        int c=compare(x.mag,y.mag);
        if (c==0) {
            return Fix(0);
        } else if(c>0) {
            r.neg=x.neg;
            r.mag = x.mag-y.mag;
        } else {
            r.neg=y.neg;
            r.mag = y.mag-x.mag;
        }
    }
    return r;
}
Fix operator-(Fix x, Fix y) {
    return x + negate(y);
}
Fix shiftLeft(Fix x) {
    Fix y;
    y.neg = x.neg;
    y.mag = shiftLeft(x.mag);
    return normalizeFix(y);
}
Fix operator>>(Fix x, int N) {
    Fix y;
    y.neg = x.neg;
    y.mag = x.mag >> N;
    return normalizeFix(y);
}
Fix FixFromFloat(float x) {
    Fix r=Fix(0);
    r.neg = (x < 0.0f);
    float ax = abs(x);

    float ip = floor(ax); //integer part
    r.mag.limb[L - 1] = (uint)ip;
    float frac = ax - ip;

    for (int i = L - 2; i >= 0; --i)
    {
        frac = frac * 4294967296.0;      // 2^32
        float di = floor(frac);
        r.mag.limb[i] = (uint)di;
        frac = frac - di;
    }

    return normalizeFix(r);
}
Fix FixFromDouble(double x) {
    Fix r=Fix(0);
    r.neg = (x < 0.0f);
    double ax = abs(x);

    double ip = floor(ax); //integer part
    r.mag.limb[L - 1] = (uint)ip;
    double frac = ax - ip;

    for (int i = L - 2; i >= 0; --i)
    {
        frac = frac * 4294967296.0;      // 2^32
        double di = floor(frac);
        r.mag.limb[i] = (uint)di;
        frac = frac - di;
    }

    return normalizeFix(r);
}
Fix operator*(Fix x, Fix y) {
    Fix r;
    BigUINT2 p = mulWide(x.mag,y.mag);
    for(int i=0;i<L;i++)
    {
        int src = i + FRAC_LIMBS;
        r.mag.limb[i] = p.limb[src];
    }
    r.neg = x.neg != y.neg;
    return r;
}
Fix square(Fix x) {
    Fix r;
    BigUINT2 p = squareWide(x.mag);
    for(int i=0;i<L;i++)
    {
        int src = i + FRAC_LIMBS;
        r.mag.limb[i] = p.limb[src];
    }
    r.neg = false;
    return r;
}

//complex number
struct FixC {
    Fix x;
    Fix y;
};
FixC operator+(FixC z1, FixC z2) {
    FixC r;
    r.x = z1.x+z2.x;
    r.y = z1.y+z2.y;
    return r;
}
// square of complex number z=x+yi
FixC square(FixC z) {
    FixC r;
    r.x = square(z.x)-square(z.y);
    r.y = shiftLeft(z.x*z.y);
    return r;
}
uint length(FixC z) {
    BigUINT r=square(z.x).mag+square(z.y).mag;
    return r.limb[L-1];
}

struct UniformBuffer {
    double2 mousePos;
    uint4 centerXMag[2]; int centerXNeg;
    uint4 centerYMag[2]; int centerYNeg;
    int shiftN;
    int iteration_count;
    float  time;
};
ConstantBuffer<UniformBuffer> ubo;

float3 palette(float t) {
    return 0.5 + 0.5*cos(6.28318*(float3(0.0, 0.33, 0.67) + t));
}

#define PERFECT_2SPOWER_RESOLUTION 1024
#define HALF_RESOLUTION 512
[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float3 color = float3(0.0);
    FixC z = FixC(0);
    Fix centerY; centerY.neg = (bool)ubo.centerYNeg;
    for (int i=0; i<2; ++i) {
        for(int j=0;j<4;++j){
            centerY.mag.limb[i*4+j] = ubo.centerYMag[i][j];
        }
    }
    Fix centerX; centerX.neg = (bool)ubo.centerXNeg;
    for (int i=0; i<2; ++i) {
        for(int j=0;j<4;++j){
            centerX.mag.limb[i*4+j] = ubo.centerXMag[i][j];
        }
    }
    FixC c; 
    c.x = FixFromFloat(vertIn.pos.x-HALF_RESOLUTION);
    c.y = FixFromFloat(vertIn.pos.y-HALF_RESOLUTION);
    c.x = c.x >> ubo.shiftN;
    c.y = c.y >> ubo.shiftN;
    c.x = centerX + c.x;
    c.y = centerY + c.y;

    int IT_COUNT=ubo.iteration_count;
    for (int i=0; i<IT_COUNT; ++i) {
        if (length(z) >= 4) {
            float v = (i+1);
            color = palette(v*0.03+.6);
            break;
        }
        z = square(z) + c;
    }

    return float4(color,1.0);
}