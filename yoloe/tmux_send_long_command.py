#!/usr/bin/env python3
import subprocess
import time
import argparse

def split_command_into_chunks(command, chunk_size=300):
    """
    Split a long command into chunks of a specified size,
    trying to avoid splitting in the middle of a word.
    """
    chunks = []
    current_chunk = ""
    
    # Split by spaces to avoid cutting words
    words = command.split()
    
    for word in words:
        if len(current_chunk) + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = word
        else:
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def send_command_to_tmux(session_name, command, delay=0.1):
    """
    Send a command to the specified tmux session with chunks to handle long commands.
    """
    print(f"Sending command to tmux session: {session_name}")
    
    # Split the command at semicolons to handle multiple commands separately
    commands = command.split(';')
    
    for i, cmd in enumerate(commands):
        # Split long commands into chunks
        chunks = split_command_into_chunks(cmd.strip())
        
        for j, chunk in enumerate(chunks):
            # Send each chunk            
            # Use subprocess to send the chunk to tmux
            subprocess.run(['tmux', 'send-keys', '-t', session_name, '--', " " +chunk])
            time.sleep(delay)  # Wait a bit between chunks
            
        # After all chunks of a command are sent, press Enter
        subprocess.run(['tmux', 'send-keys', '-t', session_name, 'Enter'])
        time.sleep(delay)  # Wait a bit between commands
    
    print("All commands sent successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a long command to a tmux session in chunks')
    parser.add_argument('session', help='The tmux session name (e.g., yoloe_predict_1e4e4970.0)')
    parser.add_argument('--command', '-c', required=True, help='The command to send (if not provided, will use the hardcoded YOLOE command)')
    parser.add_argument('--delay', '-d', type=float, default=0.1, help='Delay between sending chunks (seconds)')
    
    args = parser.parse_args()
    
    command_to_send = args.command
    
    send_command_to_tmux(args.session, command_to_send, args.delay)