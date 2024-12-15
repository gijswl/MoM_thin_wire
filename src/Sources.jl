abstract type Source end

struct VoltageSource <: Source
    V::Real
    edge::Integer
end