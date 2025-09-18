#!/usr/bin/env python3
"""
Database Migration Script
Creates and initializes the Opinion Market database with proper relationships
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from app.core.database import Base, get_database_url
from app.models import (
    user, market, trade, vote, position, order, 
    governance, advanced_markets, notification, dispute
)

def create_database():
    """Create the database and all tables"""
    try:
        # Get database URL
        database_url = get_database_url()
        print(f"🔗 Connecting to database: {database_url}")
        
        # Create engine
        engine = create_engine(database_url, echo=True)
        
        # Create all tables
        print("📋 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        
        print("✅ Database tables created successfully!")
        
        # Test the connection
        print("🧪 Testing database connection...")
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        try:
            # Test a simple query
            result = db.execute(text("SELECT 1 as test")).fetchone()
            if result and result[0] == 1:
                print("✅ Database connection test successful!")
            else:
                print("❌ Database connection test failed!")
                return False
                
        except Exception as e:
            print(f"❌ Database connection test error: {e}")
            return False
        finally:
            db.close()
        
        # Show table information
        print("\n📊 Database tables created:")
        inspector = engine.dialect.inspector(engine)
        tables = inspector.get_table_names()
        for table in sorted(tables):
            print(f"   ✅ {table}")
        
        print(f"\n🎉 Database migration completed successfully!")
        print(f"📁 Database location: {database_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def drop_database():
    """Drop all database tables (use with caution!)"""
    try:
        database_url = get_database_url()
        print(f"🗑️  Dropping database tables: {database_url}")
        
        engine = create_engine(database_url, echo=True)
        Base.metadata.drop_all(bind=engine)
        
        print("✅ Database tables dropped successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to drop database tables: {e}")
        return False

def show_database_info():
    """Show information about the current database"""
    try:
        database_url = get_database_url()
        print(f"📊 Database Information:")
        print(f"   URL: {database_url}")
        
        engine = create_engine(database_url)
        inspector = engine.dialect.inspector(engine)
        tables = inspector.get_table_names()
        
        print(f"   Tables: {len(tables)}")
        for table in sorted(tables):
            columns = inspector.get_columns(table)
            print(f"   ✅ {table} ({len(columns)} columns)")
            
            # Show foreign keys for each table
            foreign_keys = inspector.get_foreign_keys(table)
            if foreign_keys:
                for fk in foreign_keys:
                    print(f"      🔗 {fk['constrained_columns'][0]} -> {fk['referred_table']}.{fk['referred_columns'][0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to get database info: {e}")
        return False

def main():
    """Main migration function"""
    print("🚀 Opinion Market Database Migration Tool")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "create":
            success = create_database()
        elif command == "drop":
            print("⚠️  WARNING: This will drop ALL database tables!")
            confirm = input("Are you sure? Type 'yes' to continue: ")
            if confirm.lower() == 'yes':
                success = drop_database()
            else:
                print("❌ Operation cancelled.")
                success = False
        elif command == "info":
            success = show_database_info()
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: create, drop, info")
            success = False
    else:
        # Default action: create database
        success = create_database()
    
    if success:
        print("\n🎉 Migration completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
