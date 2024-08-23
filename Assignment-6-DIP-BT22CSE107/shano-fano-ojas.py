import numpy as np
import cv2
from collections import defaultdict, Counter

def shannon_fano_encoding(symbols, prefix=""):
    if len(symbols) == 1:
        return {symbols[0][0]: prefix}
    
    total_freq = sum([item[1] for item in symbols])
    cumulative_freq = 0
    split_index = 0
    
    for i, (symbol, freq) in enumerate(symbols):
        cumulative_freq += freq
        if cumulative_freq >= total_freq / 2:
            split_index = i
            break
    
    left_symbols = symbols[:split_index + 1]
    right_symbols = symbols[split_index + 1:]
    
    codebook = {}
    codebook.update(shannon_fano_encoding(left_symbols, prefix + "0"))
    codebook.update(shannon_fano_encoding(right_symbols, prefix + "1"))
    
    return codebook

def shannon_fano_channel(image_channel):
    flat_channel = image_channel.flatten()
    frequency = dict(Counter(flat_channel))
    symbols = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    
    codebook = shannon_fano_encoding(symbols)
    encoded_channel = "".join(codebook[pixel] for pixel in flat_channel)
    
    return encoded_channel, codebook

def shannon_fano_decoding(encoded_channel, codebook, shape):
    reverse_codebook = {v: k for k, v in codebook.items()}
    current_code = ""
    decoded_pixels = []
    
    for bit in encoded_channel:
        current_code += bit
        if current_code in reverse_codebook:
            decoded_pixels.append(reverse_codebook[current_code])
            current_code = ""
    
    return np.array(decoded_pixels).reshape(shape)

# Example usage
def main():
    # Load a color image
    image = cv2.imread('color_image.png')
    
    # Separate the color channels
    channels = cv2.split(image)

    encoded_channels = []
    shannon_fano_codebooks = []

    # Perform Shannon-Fano encoding on each channel
    for channel in channels:
        encoded_channel, codebook = shannon_fano_channel(channel)
        encoded_channels.append(encoded_channel)
        shannon_fano_codebooks.append(codebook)
    
    # Print the sizes
    original_size = image.size * 8
    encoded_size = sum(len(ch) for ch in encoded_channels)
    print("Original Image Size (in bits):", original_size)
    print("Encoded Image Size (in bits):", encoded_size)
    
    # Decode the image
    decoded_channels = []
    for encoded_channel, codebook, channel in zip(encoded_channels, shannon_fano_codebooks, channels):
        decoded_channel = shannon_fano_decoding(encoded_channel, codebook, channel.shape)
        decoded_channels.append(decoded_channel)
    
    # Merge the decoded channels
    decoded_image = cv2.merge(decoded_channels)

    # Save the decoded image to verify
    cv2.imwrite('decoded_image.png', decoded_image)

if __name__ == "__main__":
    main()
