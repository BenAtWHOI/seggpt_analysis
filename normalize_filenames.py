import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, help='Directory of filenames to process')
    parser.add_argument('--pattern', type=str, help='Pattern to remove from filenames')
    parser.add_argument('--replacement', type=str, default='', help='Pattern to replace into filenames')
    args = parser.parse_args()

    for filename in os.listdir(args.directory):
        if args.pattern in filename:
            new_filename = filename.replace(args.pattern, args.replacement)
            old_path = os.path.join(args.directory, filename)
            new_path = os.path.join(args.directory, new_filename)
            os.rename(old_path, new_path)

if __name__ == '__main__':
    main()