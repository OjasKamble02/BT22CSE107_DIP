import heapq
from collections import defaultdict, Counter
import numpy as np
import cv2

class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequency):
    heap = [HuffmanNode(freq, symbol) for symbol, freq in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heap[0]

def generate_huffman_codes(node, prefix="", codebook={}):
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        generate_huffman_codes(node.left, prefix + "0", codebook)
        generate_huffman_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encoding(image_channel):
    flat_channel = image_channel.flatten()
    frequency = dict(Counter(flat_channel))
    huffman_tree = build_huffman_tree(frequency)
    huffman_codes = generate_huffman_codes(huffman_tree)
    
    encoded_channel = "".join(huffman_codes[pixel] for pixel in flat_channel)
    
    return encoded_channel, huffman_codes

def huffman_decoding(encoded_channel, huffman_codes, shape):
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    current_code = ""
    decoded_pixels = []
    
    for bit in encoded_channel:
        current_code += bit
        if current_code in reverse_codes:
            decoded_pixels.append(reverse_codes[current_code])
            current_code = ""
    
    return np.array(decoded_pixels).reshape(shape)

# Example usage
def main():
    # Load a color image
    image = cv2.imread('color_image.png')
    
    # Separate the color channels
    channels = cv2.split(image)

    encoded_channels = []
    huffman_codebooks = []

    # Perform Huffman encoding on each channel
    for channel in channels:
        encoded_channel, huffman_codes = huffman_encoding(channel)
        encoded_channels.append(encoded_channel)
        huffman_codebooks.append(huffman_codes)
    
    # Print the sizes
    original_size = image.size * 8
    encoded_size = sum(len(ch) for ch in encoded_channels)
    print("Original Image Size (in bits):", original_size)
    print("Encoded Image Size (in bits):", encoded_size)
    
    # Decode the image
    decoded_channels = []
    for encoded_channel, huffman_codes, channel in zip(encoded_channels, huffman_codebooks, channels):
        decoded_channel = huffman_decoding(encoded_channel, huffman_codes, channel.shape)
        decoded_channels.append(decoded_channel)
    
    # Merge the decoded channels
    decoded_image = cv2.merge(decoded_channels)

    # Save the decoded image to verify
    cv2.imwrite('decoded_image.png', decoded_image)

if __name__ == "__main__":
    main()
