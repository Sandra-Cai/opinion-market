#!/usr/bin/env python3
"""
Database Schema Validation Script
Validates that all database relationships are properly configured
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from app.core.database import Base, get_database_url
from app.models import (
    user, market, trade, vote, position, order, 
    governance, advanced_markets, notification, dispute
)

def validate_foreign_keys():
    """Validate all foreign key relationships"""
    print("🔍 Validating Foreign Key Relationships...")
    
    try:
        # Get database URL
        database_url = get_database_url()
        print(f"🔗 Connecting to database: {database_url}")
        
        # Create engine
        engine = create_engine(database_url)
        inspector = inspect(engine)
        
        # Get all tables
        tables = inspector.get_table_names()
        print(f"📋 Found {len(tables)} tables: {', '.join(sorted(tables))}")
        
        # Check foreign keys for each table
        foreign_key_issues = []
        
        for table_name in sorted(tables):
            print(f"\n🔍 Checking table: {table_name}")
            
            # Get columns
            columns = inspector.get_columns(table_name)
            print(f"   Columns: {len(columns)}")
            
            # Get foreign keys
            foreign_keys = inspector.get_foreign_keys(table_name)
            print(f"   Foreign Keys: {len(foreign_keys)}")
            
            for fk in foreign_keys:
                constrained_columns = fk['constrained_columns']
                referred_table = fk['referred_table']
                referred_columns = fk['referred_columns']
                
                print(f"      🔗 {constrained_columns[0]} -> {referred_table}.{referred_columns[0]}")
                
                # Check if referred table exists
                if referred_table not in tables:
                    issue = f"Table {table_name}: Foreign key {constrained_columns[0]} references non-existent table {referred_table}"
                    foreign_key_issues.append(issue)
                    print(f"         ❌ {issue}")
                else:
                    print(f"         ✅ Referenced table exists")
        
        if foreign_key_issues:
            print(f"\n❌ Found {len(foreign_key_issues)} foreign key issues:")
            for issue in foreign_key_issues:
                print(f"   - {issue}")
            return False
        else:
            print(f"\n✅ All foreign key relationships are valid!")
            return True
            
    except Exception as e:
        print(f"❌ Error validating foreign keys: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_sqlalchemy_relationships():
    """Validate SQLAlchemy model relationships"""
    print("\n🔍 Validating SQLAlchemy Model Relationships...")
    
    try:
        # Get database URL
        database_url = get_database_url()
        engine = create_engine(database_url)
        
        # Create session
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        try:
            # Test basic queries to ensure relationships work
            print("🧪 Testing model relationships...")
            
            # Test User model
            print("   Testing User model...")
            from app.models.user import User
            user_count = db.query(User).count()
            print(f"      Users in database: {user_count}")
            
            # Test Market model
            print("   Testing Market model...")
            from app.models.market import Market
            market_count = db.query(Market).count()
            print(f"      Markets in database: {market_count}")
            
            # Test Trade model
            print("   Testing Trade model...")
            from app.models.trade import Trade
            trade_count = db.query(Trade).count()
            print(f"      Trades in database: {trade_count}")
            
            # Test Vote model
            print("   Testing Vote model...")
            from app.models.vote import Vote
            vote_count = db.query(Vote).count()
            print(f"      Votes in database: {vote_count}")
            
            # Test relationship queries
            print("   Testing relationship queries...")
            
            # Test User -> Markets relationship
            if user_count > 0:
                user_with_markets = db.query(User).first()
                try:
                    markets_created = user_with_markets.markets_created
                    print(f"      ✅ User.markets_created relationship works")
                except Exception as e:
                    print(f"      ❌ User.markets_created relationship failed: {e}")
            
            # Test Market -> Trades relationship
            if market_count > 0:
                market_with_trades = db.query(Market).first()
                try:
                    trades = market_with_trades.trades
                    print(f"      ✅ Market.trades relationship works")
                except Exception as e:
                    print(f"      ❌ Market.trades relationship failed: {e}")
            
            # Test Market -> Votes relationship
            if market_count > 0:
                market_with_votes = db.query(Market).first()
                try:
                    votes = market_with_votes.votes
                    print(f"      ✅ Market.votes relationship works")
                except Exception as e:
                    print(f"      ❌ Market.votes relationship failed: {e}")
            
            print("✅ All SQLAlchemy relationships are working!")
            return True
            
        except Exception as e:
            print(f"❌ Error testing SQLAlchemy relationships: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error validating SQLAlchemy relationships: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_database_integrity():
    """Validate database integrity"""
    print("\n🔍 Validating Database Integrity...")
    
    try:
        # Get database URL
        database_url = get_database_url()
        engine = create_engine(database_url)
        
        # Create session
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        try:
            # Test database connection
            result = db.execute(text("SELECT 1 as test")).fetchone()
            if result and result[0] == 1:
                print("✅ Database connection test successful")
            else:
                print("❌ Database connection test failed")
                return False
            
            # Test table existence
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            expected_tables = [
                'users', 'markets', 'trades', 'votes', 'positions', 'orders',
                'market_disputes', 'dispute_votes', 'notifications', 
                'notification_preferences', 'governance_proposals', 
                'governance_votes', 'governance_tokens', 'futures_contracts',
                'futures_positions', 'options_contracts', 'options_positions',
                'conditional_markets', 'spread_markets'
            ]
            
            missing_tables = []
            for table in expected_tables:
                if table not in tables:
                    missing_tables.append(table)
            
            if missing_tables:
                print(f"❌ Missing tables: {', '.join(missing_tables)}")
                return False
            else:
                print("✅ All expected tables exist")
            
            # Test basic queries on each table
            print("🧪 Testing basic queries on all tables...")
            
            for table in sorted(tables):
                try:
                    result = db.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                    count = result[0] if result else 0
                    print(f"   ✅ {table}: {count} records")
                except Exception as e:
                    print(f"   ❌ {table}: Error - {e}")
                    return False
            
            print("✅ Database integrity validation successful!")
            return True
            
        except Exception as e:
            print(f"❌ Error validating database integrity: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error validating database integrity: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_schema_report():
    """Generate a comprehensive schema report"""
    print("\n📊 Generating Schema Report...")
    
    try:
        # Get database URL
        database_url = get_database_url()
        engine = create_engine(database_url)
        inspector = inspect(engine)
        
        # Get all tables
        tables = inspector.get_table_names()
        
        report = {
            "database_url": database_url,
            "total_tables": len(tables),
            "tables": {}
        }
        
        for table_name in sorted(tables):
            columns = inspector.get_columns(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            
            table_info = {
                "columns": len(columns),
                "foreign_keys": len(foreign_keys),
                "column_details": [],
                "foreign_key_details": []
            }
            
            # Column details
            for column in columns:
                table_info["column_details"].append({
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                    "primary_key": column.get("primary_key", False)
                })
            
            # Foreign key details
            for fk in foreign_keys:
                table_info["foreign_key_details"].append({
                    "constrained_columns": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"]
                })
            
            report["tables"][table_name] = table_info
        
        # Save report to file
        import json
        from datetime import datetime
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"database_schema_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Schema report saved to: {filename}")
        print(f"📊 Report summary:")
        print(f"   Total tables: {report['total_tables']}")
        print(f"   Total columns: {sum(t['columns'] for t in report['tables'].values())}")
        print(f"   Total foreign keys: {sum(t['foreign_keys'] for t in report['tables'].values())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating schema report: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function"""
    print("🚀 Database Schema Validation Tool")
    print("=" * 50)
    
    # Run all validations
    validations = [
        ("Foreign Key Relationships", validate_foreign_keys),
        ("SQLAlchemy Model Relationships", validate_sqlalchemy_relationships),
        ("Database Integrity", validate_database_integrity),
        ("Schema Report Generation", generate_schema_report)
    ]
    
    results = []
    
    for name, validation_func in validations:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = validation_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All validations passed! Database schema is valid.")
        return True
    else:
        print("⚠️  Some validations failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
