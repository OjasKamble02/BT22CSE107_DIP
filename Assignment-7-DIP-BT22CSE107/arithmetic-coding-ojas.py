import numpy as np
import cv2
from collections import Counter, defaultdict

class ArithmeticCoder:
    def __init__(self, frequencies):
        self.frequencies = frequencies
        self.total = sum(frequencies.values())
        self.low = 0.0
        self.high = 1.0
        self.precision = 32
        self.mask = (1 << self.precision) - 1
        self.scale3 = 1 << (self.precision - 1)
        self.encoded_bits = []

    def encode_symbol(self, symbol):
        low_range = sum(self.frequencies[s] for s in self.frequencies if s < symbol)
        high_range = low_range + self.frequencies[symbol]

        range_ = self.high - self.low
        self.high = self.low + range_ * (high_range / self.total)
        self.low = self.low + range_ * (low_range / self.total)

        while True:
            if self.high <= 0.5:
                self.encoded_bits.append(0)
                self.low *= 2
                self.high *= 2
            elif self.low >= 0.5:
                self.encoded_bits.append(1)
                self.low = 2 * (self.low - 0.5)
                self.high = 2 * (self.high - 0.5)
            elif self.low >= 0.25 and self.high <= 0.75:
                self.encoded_bits.append(0)
                self.low = 2 * (self.low - 0.25)
                self.high = 2 * (self.high - 0.25)
            else:
                break

    def finalize(self):
        if self.low < 0.25:
            self.encoded_bits.append(0)
        else:
            self.encoded_bits.append(1)
        
        return self.encoded_bits

    def encode(self, data):
        for symbol in data:
            self.encode_symbol(symbol)
        return self.finalize()

    def decode(self, encoded_bits, data_len):
        value = 0.0
        low = 0.0
        high = 1.0
        range_ = high - low

        decoded = []
        for i in range(data_len):
            value = (value * 2 + encoded_bits[i]) / range_
            low_range = 0
            high_range = 0
            for symbol, freq in self.frequencies.items():
                high_range = low_range + freq
                if low_range / self.total <= value < high_range / self.total:
                    decoded.append(symbol)
                    low += range_ * (low_range / self.total)
                    high = low + range_ * (freq / self.total)
                    range_ = high - low
                    break
                low_range = high_range

        return np.array(decoded)

# Example usage
def main():
    # Load a color image
    image = cv2.imread('color_image.png')
    
    # Separate the color channels
    channels = cv2.split(image)

    encoded_channels = []
    arithmetic_coders = []
    
    # Perform Arithmetic encoding on each channel
    for channel in channels:
        flat_channel = channel.flatten()
        frequencies = dict(Counter(flat_channel))
        coder = ArithmeticCoder(frequencies)
        encoded_channel = coder.encode(flat_channel)
        encoded_channels.append(encoded_channel)
        arithmetic_coders.append(coder)
    
    # Print the sizes
    original_size = image.size * 8
    encoded_size = sum(len(ch) for ch in encoded_channels)
    print("Original Image Size (in bits):", original_size)
    print("Encoded Image Size (in bits):", encoded_size)
    
    # Decode the image
    decoded_channels = []
    for encoded_channel, coder, channel in zip(encoded_channels, arithmetic_coders, channels):
        decoded_channel = coder.decode(encoded_channel, channel.size)
        decoded_channels.append(decoded_channel.reshape(channel.shape))
    
    # Merge the decoded channels
    decoded_image = cv2.merge(decoded_channels)

    # Save the decoded image to verify
    cv2.imwrite('decoded_image.png', decoded_image)

if __name__ == "__main__":
    main()
