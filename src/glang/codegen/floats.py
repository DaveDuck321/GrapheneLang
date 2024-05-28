from decimal import Decimal

from glang.codegen.user_facing_errors import (
    InvalidFloatLiteralPrecision,
    InvalidFloatLiteralTooLarge,
)

EXPONENT_BIT_TABLE = {16: 5, 32: 8, 64: 11, 128: 15}
FRACTION_BIT_TABLE = {16: 10, 32: 23, 64: 52, 128: 112}


def fraction_to_binary_list(
    number: Decimal, precision_in_bits: int, list_length: int
) -> list[bool]:
    assert number <= 1
    without_decimal_place = int(number * 2**precision_in_bits)

    result = [False] * (list_length - precision_in_bits)
    for _ in range(precision_in_bits):
        result.append((without_decimal_place & 1) == 1)
        without_decimal_place >>= 1

    assert without_decimal_place == 0
    return result


def float_literal_to_exact_hex(input_float_str: str, float_size_in_bits: int) -> str:
    float_str = input_float_str.lower()

    # We support both exponential and decimal notation
    # First simplify the algorithm by inserting the implicit exponent
    if "e" not in float_str:
        float_str += "e0"

    # Now trim the sign (if any)
    # IEEE floats have symmetric precision so both cases are identical
    is_negative = float_str[0] == "-"
    float_str = float_str.lstrip("+-")

    mantissa_str_10, exponent_str_10 = float_str.split("e")

    # The exponent is an integer, python can give us an exact representation
    exponent_10 = int(exponent_str_10)

    # The mantissa may contain a decimal place, adjust the exponent until it is an integer
    if "." in mantissa_str_10:
        decimal_point_index = mantissa_str_10.index(".")

        mantissa_str_10 = mantissa_str_10.replace(".", "")
        exponent_10 += decimal_point_index - len(mantissa_str_10)

    mantissa_10 = int(mantissa_str_10)

    if mantissa_10 == 0:
        # Note: I'm discarding '-0.0' here... I'm not sure if its worth propagating
        # max(...) is due to LLVM weirdness
        return "0x" + "00" * (max(float_size_in_bits, 64) // 8)

    exponent_natural_bits = EXPONENT_BIT_TABLE[float_size_in_bits]
    fraction_natural_bits = FRACTION_BIT_TABLE[float_size_in_bits]
    assert 1 + exponent_natural_bits + fraction_natural_bits == float_size_in_bits

    # I should probably look at an official algorithm here...
    # Its a fun problem tho, I'll just derive something
    #  See if you can work out where my magic numbers came from :-D
    exponent_2 = 3 * exponent_10
    mantissa_2 = mantissa_10 * ((Decimal(5) / Decimal(4)) ** exponent_10)

    # Now normalize float eg. 1.something * 10_2**n
    if mantissa_2 > 1:
        while mantissa_2 >= 2:
            mantissa_2 /= 2
            exponent_2 += 1
    else:
        while mantissa_2 < 1:
            mantissa_2 *= 2
            exponent_2 -= 1

    # Round away from zero if needed
    shifted_number = mantissa_2 * 2**fraction_natural_bits
    should_round = (shifted_number - int(shifted_number)) > 0.5
    if should_round:
        mantissa_2 += Decimal(2) ** -fraction_natural_bits
        if mantissa_2 >= 2:
            mantissa_2 /= 2
            exponent_2 += 1

    # Is this number in the range we can represent?

    # The exponent cannot be all zeros (subnormal) or all ones (infinity)
    #   We support subnormal floating point numbers but NOT as literals
    max_exponent = 2 ** (exponent_natural_bits - 1) - 1
    min_exponent = -(max_exponent - 1)

    if exponent_2 > max_exponent:
        least_significant_bit = max_exponent - fraction_natural_bits
        min_unrepresentable = 2 ** (max_exponent + 1)
        max_representable = min_unrepresentable - 2**least_significant_bit
        max_representable_with_rounding = (min_unrepresentable + max_representable) // 2

        raise InvalidFloatLiteralTooLarge(
            f"f{float_size_in_bits}",
            input_float_str,
            max_representable_with_rounding,
            min_unrepresentable,
        )

    if exponent_2 < min_exponent:
        # This is a *small* lie: we could always give a subnormal value
        raise InvalidFloatLiteralPrecision(f"f{float_size_in_bits}", input_float_str)

    # Convert the exponent into a binary list (ik, slow right?)
    # Performance of the normalization step is likely the bottleneck regardless

    # NOTE: LLVM requires 16 and 32 bit floats to be represented using the 64 format!!
    exponent_llvm_bits = EXPONENT_BIT_TABLE[max(float_size_in_bits, 64)]
    fraction_llvm_bits = FRACTION_BIT_TABLE[max(float_size_in_bits, 64)]

    # The exponent is stored in 1s complement
    exponent_2 += 2 ** (exponent_llvm_bits - 1) - 1

    exponent_bit_list: list[bool] = []
    while len(exponent_bit_list) < exponent_llvm_bits:
        exponent_bit_list.append((exponent_2 & 1) == 1)
        exponent_2 >>= 1

    assert exponent_2 == 0

    # The leading 1 is implicit for normalized floats
    mantissa_2 -= 1

    fraction_bit_list = fraction_to_binary_list(
        mantissa_2, fraction_natural_bits, fraction_llvm_bits
    )

    # Combine everything
    float_bit_list = [*fraction_bit_list, *exponent_bit_list, is_negative]

    # We've finished!
    result = sum((bit * (2**i) for i, bit in enumerate(float_bit_list)))
    return f"0x{result:0{float_size_in_bits // 4}x}"
