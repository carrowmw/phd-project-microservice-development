#!/bin/bash

# Set the source directory (your project directory)
SOURCE_DIR="/Users/administrator/Code/python/phd-project/"

# Set the destination directory (your OneDrive folder)
DEST_DIR="/Users/administrator/OneDrive - Newcastle University/Backups/Results/"

# Create a timestamp for the backup folder
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="$DEST_DIR/backup_$TIMESTAMP"

# Create the backup directory
mkdir -p "$BACKUP_DIR"

# List of folders to backup
FOLDERS_TO_BACKUP=(
  "mlruns"
  "phd_package/data/pipeline/test_metrics"
  # Add more folders as needed
)

# Function to backup a folder
backup_folder() {
  local folder="$1"
  echo "Backing up $folder..."
  rsync -a --info=progress2 "$SOURCE_DIR/$folder" "$BACKUP_DIR/"
}

# Perform the backup
for folder in "${FOLDERS_TO_BACKUP[@]}"; do
  backup_folder "$folder"
done

echo "Backup completed successfully to $BACKUP_DIR"