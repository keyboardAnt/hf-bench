#!/bin/bash

# Function to convert bytes to human-readable format
convert_size() {
    local size=$1
    echo $(numfmt --to=iec-i --suffix=B $size)
}

# Store initial sizes
echo "=== Initial Storage Status ==="
echo "Checking home directory usage..."
total_before=$(df -B1 $HOME | awk 'NR==2 {print $3}') || { echo "Error checking home directory size"; exit 1; }

echo "Checking cache size..."
cache_before=$(du -sb ~/.cache 2>/dev/null | awk '{print $1}') || echo "0"

echo "Checking conda size..."
conda_before=$(du -sb ~/.conda 2>/dev/null | awk '{print $1}') || echo "0"

echo "Checking trash size..."
trash_before=$(du -sb ~/.local/share/Trash 2>/dev/null | awk '{print $1}') || echo "0"

echo "Home directory usage: $(convert_size $total_before)"
echo "Cache size: $(convert_size $cache_before)"
echo "Conda size: $(convert_size $conda_before)"
echo "Trash size: $(convert_size $trash_before)"

# Confirm before proceeding
read -p "Proceed with cleanup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Perform cleanup
echo -e "\n=== Performing Cleanup ==="

# Clear cache
echo "Clearing cache..."
rm -rf ~/.cache/* 2>/dev/null

# Clear conda
echo "Clearing conda cache..."
rm -rf ~/.conda/* 2>/dev/null

# Empty trash
echo "Emptying trash..."
rm -rf ~/.local/share/Trash/info/ ~/.local/share/Trash/files/ 2>/dev/null

# Calculate space freed
total_after=$(df -B1 $HOME | awk 'NR==2 {print $3}')
cache_after=$(du -sb ~/.cache 2>/dev/null | awk '{print $1}')
conda_after=$(du -sb ~/.conda 2>/dev/null | awk '{print $1}')
trash_after=$(du -sb ~/.local/share/Trash 2>/dev/null | awk '{print $1}')

echo -e "\n=== Storage Freed ==="
echo "Cache: $(convert_size $((cache_before - cache_after)))"
echo "Conda: $(convert_size $((conda_before - conda_after)))"
echo "Trash: $(convert_size $((trash_before - trash_after)))"
echo "Total space freed: $(convert_size $((total_before - total_after)))" 