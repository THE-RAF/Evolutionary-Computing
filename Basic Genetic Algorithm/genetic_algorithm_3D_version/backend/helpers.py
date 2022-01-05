def translate_bits_to_float(bit_string, interval):
    step = (interval[1] - interval[0]) / (2**len(bit_string))
    x = step * sum([int(bit) * 2**i for i, bit in enumerate(bit_string)])

    return x
