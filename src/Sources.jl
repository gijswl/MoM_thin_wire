abstract type Source end

# A delta-gap voltage source
struct VoltageSource <: Source
    V::Real
    edge::Integer
end

# Current source not implemented