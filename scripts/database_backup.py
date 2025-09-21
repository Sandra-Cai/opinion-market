#!/usr/bin/env python3
"""
Database Backup and Restore System
Provides comprehensive database backup and restore functionality
"""

import os
import sys
import shutil
import sqlite3
import json
import gzip
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.database import get_database_url
from app.models import (
    user, market, trade, vote, position, order, 
    governance, advanced_markets, notification, dispute
)

class DatabaseBackupManager:
    """Manages database backup and restore operations"""
    
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.database_url = get_database_url()
        self.db_path = self._extract_db_path()
    
    def _extract_db_path(self) -> str:
        """Extract database file path from URL"""
        if self.database_url.startswith("sqlite:///"):
            return self.database_url.replace("sqlite:///", "")
        else:
            raise ValueError(f"Unsupported database URL: {self.database_url}")
    
    def create_backup(self, name: Optional[str] = None, compress: bool = True) -> str:
        """Create a database backup"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        
        # Generate backup name
        if name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            name = f"backup_{timestamp}"
        
        backup_file = self.backup_dir / f"{name}.db"
        compressed_file = self.backup_dir / f"{name}.db.gz"
        
        print(f"🔄 Creating backup: {name}")
        print(f"   Source: {self.db_path}")
        print(f"   Destination: {backup_file}")
        
        # Copy database file
        shutil.copy2(self.db_path, backup_file)
        
        # Compress if requested
        if compress:
            print(f"   Compressing backup...")
            with open(backup_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove uncompressed file
            backup_file.unlink()
            backup_file = compressed_file
        
        # Create backup metadata
        metadata = {
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "database_url": self.database_url,
            "database_size": os.path.getsize(self.db_path),
            "backup_size": os.path.getsize(backup_file),
            "compressed": compress,
            "tables": self._get_table_info()
        }
        
        metadata_file = self.backup_dir / f"{name}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Backup created successfully: {backup_file}")
        print(f"   Size: {metadata['backup_size']} bytes")
        print(f"   Tables: {len(metadata['tables'])}")
        
        return str(backup_file)
    
    def restore_backup(self, name: str, create_backup: bool = True) -> bool:
        """Restore database from backup"""
        backup_file = self.backup_dir / f"{name}.db"
        compressed_file = self.backup_dir / f"{name}.db.gz"
        metadata_file = self.backup_dir / f"{name}.json"
        
        # Check if backup exists
        if not backup_file.exists() and not compressed_file.exists():
            raise FileNotFoundError(f"Backup not found: {name}")
        
        # Load metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"📋 Backup metadata:")
            print(f"   Created: {metadata.get('created_at', 'Unknown')}")
            print(f"   Size: {metadata.get('backup_size', 'Unknown')} bytes")
            print(f"   Tables: {len(metadata.get('tables', []))}")
        else:
            print("⚠️  No metadata file found")
        
        # Create backup of current database if requested
        if create_backup and os.path.exists(self.db_path):
            print("🔄 Creating backup of current database...")
            current_backup = self.create_backup(f"pre_restore_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
            print(f"   Current database backed up to: {current_backup}")
        
        # Determine source file
        source_file = compressed_file if compressed_file.exists() else backup_file
        
        print(f"🔄 Restoring database from: {source_file}")
        print(f"   Target: {self.db_path}")
        
        # Restore database
        if source_file.suffix == '.gz':
            # Decompress and restore
            with gzip.open(source_file, 'rb') as f_in:
                with open(self.db_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            # Direct copy
            shutil.copy2(source_file, self.db_path)
        
        # Verify restoration
        if self._verify_database():
            print("✅ Database restored successfully!")
            return True
        else:
            print("❌ Database restoration failed!")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        backups = []
        
        for file in self.backup_dir.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    metadata = json.load(f)
                backups.append(metadata)
            except Exception as e:
                print(f"⚠️  Error reading metadata for {file}: {e}")
        
        # Sort by creation date
        backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return backups
    
    def delete_backup(self, name: str) -> bool:
        """Delete a backup"""
        files_to_delete = [
            self.backup_dir / f"{name}.db",
            self.backup_dir / f"{name}.db.gz",
            self.backup_dir / f"{name}.json"
        ]
        
        deleted_files = []
        for file in files_to_delete:
            if file.exists():
                file.unlink()
                deleted_files.append(file.name)
        
        if deleted_files:
            print(f"✅ Deleted backup: {name}")
            print(f"   Files removed: {', '.join(deleted_files)}")
            return True
        else:
            print(f"❌ Backup not found: {name}")
            return False
    
    def _get_table_info(self) -> List[Dict[str, Any]]:
        """Get information about database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            table_info = []
            for (table_name,) in tables:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                table_info.append({
                    "name": table_name,
                    "row_count": row_count,
                    "columns": len(columns)
                })
            
            conn.close()
            return table_info
            
        except Exception as e:
            print(f"⚠️  Error getting table info: {e}")
            return []
    
    def _verify_database(self) -> bool:
        """Verify that the restored database is valid"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if we can query the database
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            conn.close()
            
            if len(tables) > 0:
                print(f"   ✅ Database verification successful: {len(tables)} tables found")
                return True
            else:
                print("   ❌ Database verification failed: No tables found")
                return False
                
        except Exception as e:
            print(f"   ❌ Database verification failed: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Clean up old backups, keeping only the most recent ones"""
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            print(f"📋 No cleanup needed: {len(backups)} backups (limit: {keep_count})")
            return 0
        
        # Delete old backups
        deleted_count = 0
        for backup in backups[keep_count:]:
            if self.delete_backup(backup['name']):
                deleted_count += 1
        
        print(f"🧹 Cleanup completed: {deleted_count} old backups removed")
        return deleted_count
    
    def export_to_sql(self, name: Optional[str] = None) -> str:
        """Export database to SQL file"""
        if name is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            name = f"export_{timestamp}"
        
        sql_file = self.backup_dir / f"{name}.sql"
        
        print(f"🔄 Exporting database to SQL: {sql_file}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            with open(sql_file, 'w') as f:
                # Export schema
                f.write("-- Database Schema Export\n")
                f.write(f"-- Generated: {datetime.utcnow().isoformat()}\n\n")
                
                # Get all tables
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                for (table_name,) in tables:
                    # Get table schema
                    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                    schema = cursor.fetchone()
                    if schema and schema[0]:
                        f.write(f"{schema[0]};\n\n")
                    
                    # Get table data
                    cursor.execute(f"SELECT * FROM {table_name}")
                    rows = cursor.fetchall()
                    
                    if rows:
                        # Get column names
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = [col[1] for col in cursor.fetchall()]
                        
                        f.write(f"-- Data for table {table_name}\n")
                        for row in rows:
                            values = []
                            for value in row:
                                if value is None:
                                    values.append("NULL")
                                elif isinstance(value, str):
                                    values.append(f"'{value.replace("'", "''")}'")
                                else:
                                    values.append(str(value))
                            
                            f.write(f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)});\n")
                        f.write("\n")
            
            conn.close()
            
            print(f"✅ SQL export completed: {sql_file}")
            print(f"   Size: {os.path.getsize(sql_file)} bytes")
            
            return str(sql_file)
            
        except Exception as e:
            print(f"❌ SQL export failed: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Database Backup and Restore Tool")
    parser.add_argument("action", choices=["backup", "restore", "list", "delete", "cleanup", "export"],
                       help="Action to perform")
    parser.add_argument("--name", help="Backup name")
    parser.add_argument("--backup-dir", default="backups", help="Backup directory")
    parser.add_argument("--no-compress", action="store_true", help="Don't compress backups")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup before restore")
    parser.add_argument("--keep", type=int, default=10, help="Number of backups to keep during cleanup")
    
    args = parser.parse_args()
    
    try:
        manager = DatabaseBackupManager(args.backup_dir)
        
        if args.action == "backup":
            backup_file = manager.create_backup(args.name, not args.no_compress)
            print(f"🎉 Backup created: {backup_file}")
            
        elif args.action == "restore":
            if not args.name:
                print("❌ Backup name required for restore")
                sys.exit(1)
            
            success = manager.restore_backup(args.name, not args.no_backup)
            if success:
                print("🎉 Database restored successfully!")
            else:
                print("❌ Database restore failed!")
                sys.exit(1)
                
        elif args.action == "list":
            backups = manager.list_backups()
            if backups:
                print("📋 Available Backups:")
                print("-" * 80)
                for backup in backups:
                    print(f"Name: {backup['name']}")
                    print(f"Created: {backup.get('created_at', 'Unknown')}")
                    print(f"Size: {backup.get('backup_size', 'Unknown')} bytes")
                    print(f"Tables: {len(backup.get('tables', []))}")
                    print(f"Compressed: {backup.get('compressed', 'Unknown')}")
                    print("-" * 80)
            else:
                print("📋 No backups found")
                
        elif args.action == "delete":
            if not args.name:
                print("❌ Backup name required for delete")
                sys.exit(1)
            
            success = manager.delete_backup(args.name)
            if not success:
                sys.exit(1)
                
        elif args.action == "cleanup":
            deleted_count = manager.cleanup_old_backups(args.keep)
            print(f"🎉 Cleanup completed: {deleted_count} backups removed")
            
        elif args.action == "export":
            sql_file = manager.export_to_sql(args.name)
            print(f"🎉 SQL export completed: {sql_file}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

