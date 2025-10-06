"""
Smart Contract Management Engine
Advanced smart contract deployment, interaction, and monitoring
"""

import asyncio
import logging
import time
import json
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ContractType(Enum):
    """Contract types"""
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    DEFI_PROTOCOL = "defi_protocol"
    GOVERNANCE = "governance"
    ORACLE = "oracle"
    BRIDGE = "bridge"
    STAKING = "staking"
    VESTING = "vesting"
    MULTISIG = "multisig"

class ContractStatus(Enum):
    """Contract status"""
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    VERIFIED = "verified"
    FAILED = "failed"
    PAUSED = "paused"
    UPGRADED = "upgraded"

class ContractFunction(Enum):
    """Contract functions"""
    TRANSFER = "transfer"
    APPROVE = "approve"
    MINT = "mint"
    BURN = "burn"
    SWAP = "swap"
    STAKE = "stake"
    UNSTAKE = "unstake"
    VOTE = "vote"
    EXECUTE = "execute"
    UPGRADE = "upgrade"

@dataclass
class SmartContract:
    """Smart contract"""
    contract_id: str
    name: str
    contract_type: ContractType
    blockchain: str
    address: str
    abi: Dict[str, Any]
    bytecode: str
    source_code: str
    compiler_version: str
    status: ContractStatus
    deployed_at: datetime
    gas_used: int
    gas_price: Decimal
    deployer_address: str
    constructor_args: List[Any] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    is_verified: bool = False
    verification_tx: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContractInteraction:
    """Contract interaction"""
    interaction_id: str
    contract_id: str
    function_name: str
    function_type: ContractFunction
    caller_address: str
    parameters: Dict[str, Any]
    gas_limit: int
    gas_price: Decimal
    value: Decimal
    tx_hash: str
    block_number: int
    status: str
    gas_used: int
    timestamp: datetime
    result: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContractEvent:
    """Contract event"""
    event_id: str
    contract_id: str
    event_name: str
    block_number: int
    transaction_hash: str
    log_index: int
    topics: List[str]
    data: str
    decoded_data: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContractTemplate:
    """Contract template"""
    template_id: str
    name: str
    contract_type: ContractType
    description: str
    source_code: str
    abi: Dict[str, Any]
    bytecode: str
    constructor_params: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    gas_estimate: int
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class SmartContractEngine:
    """Advanced Smart Contract Management Engine"""
    
    def __init__(self):
        self.engine_id = f"smart_contract_{secrets.token_hex(8)}"
        self.is_running = False
        
        # Contract data
        self.contracts: Dict[str, SmartContract] = {}
        self.interactions: List[ContractInteraction] = []
        self.events: List[ContractEvent] = []
        self.templates: Dict[str, ContractTemplate] = {}
        
        # Contract configurations
        self.contract_configs = {
            ContractType.ERC20: {
                "gas_estimate": 2000000,
                "required_functions": ["transfer", "approve", "balanceOf", "totalSupply"],
                "required_events": ["Transfer", "Approval"]
            },
            ContractType.ERC721: {
                "gas_estimate": 3000000,
                "required_functions": ["transferFrom", "approve", "ownerOf", "safeTransferFrom"],
                "required_events": ["Transfer", "Approval", "ApprovalForAll"]
            },
            ContractType.DEFI_PROTOCOL: {
                "gas_estimate": 5000000,
                "required_functions": ["deposit", "withdraw", "swap", "stake"],
                "required_events": ["Deposit", "Withdraw", "Swap", "Stake"]
            },
            ContractType.GOVERNANCE: {
                "gas_estimate": 4000000,
                "required_functions": ["propose", "vote", "execute", "queue"],
                "required_events": ["ProposalCreated", "VoteCast", "ProposalExecuted"]
            }
        }
        
        # Blockchain configurations
        self.blockchain_configs = {
            "ethereum": {
                "chain_id": 1,
                "gas_price": Decimal("20"),  # gwei
                "gas_limit": 30000000,
                "block_time": 12  # seconds
            },
            "polygon": {
                "chain_id": 137,
                "gas_price": Decimal("30"),  # gwei
                "gas_limit": 30000000,
                "block_time": 2  # seconds
            },
            "bsc": {
                "chain_id": 56,
                "gas_price": Decimal("5"),  # gwei
                "gas_limit": 30000000,
                "block_time": 3  # seconds
            },
            "avalanche": {
                "chain_id": 43114,
                "gas_price": Decimal("25"),  # gwei
                "gas_limit": 30000000,
                "block_time": 2  # seconds
            }
        }
        
        # Processing tasks
        self.contract_monitoring_task: Optional[asyncio.Task] = None
        self.event_processing_task: Optional[asyncio.Task] = None
        self.verification_task: Optional[asyncio.Task] = None
        self.performance_monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.deployment_stats: Dict[str, List[float]] = {}
        self.interaction_stats: Dict[str, List[float]] = {}
        self.gas_usage_stats: Dict[str, List[int]] = {}
        
        logger.info(f"Smart Contract Engine {self.engine_id} initialized")

    async def start_smart_contract_engine(self):
        """Start the smart contract engine"""
        if self.is_running:
            return
        
        logger.info("Starting Smart Contract Engine...")
        
        # Initialize contract data
        await self._initialize_contract_templates()
        await self._initialize_existing_contracts()
        
        # Start processing tasks
        self.is_running = True
        
        self.contract_monitoring_task = asyncio.create_task(self._contract_monitoring_loop())
        self.event_processing_task = asyncio.create_task(self._event_processing_loop())
        self.verification_task = asyncio.create_task(self._verification_loop())
        self.performance_monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
        
        logger.info("Smart Contract Engine started")

    async def stop_smart_contract_engine(self):
        """Stop the smart contract engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping Smart Contract Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.contract_monitoring_task,
            self.event_processing_task,
            self.verification_task,
            self.performance_monitoring_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Smart Contract Engine stopped")

    async def _initialize_contract_templates(self):
        """Initialize contract templates"""
        try:
            # Create contract templates
            templates = [
                {
                    "name": "Standard ERC20 Token",
                    "contract_type": ContractType.ERC20,
                    "description": "Standard ERC20 token implementation",
                    "gas_estimate": 2000000
                },
                {
                    "name": "Standard ERC721 NFT",
                    "contract_type": ContractType.ERC721,
                    "description": "Standard ERC721 NFT implementation",
                    "gas_estimate": 3000000
                },
                {
                    "name": "DeFi Staking Contract",
                    "contract_type": ContractType.STAKING,
                    "description": "DeFi staking contract with rewards",
                    "gas_estimate": 4000000
                },
                {
                    "name": "Governance Contract",
                    "contract_type": ContractType.GOVERNANCE,
                    "description": "DAO governance contract",
                    "gas_estimate": 5000000
                },
                {
                    "name": "Token Vesting Contract",
                    "contract_type": ContractType.VESTING,
                    "description": "Token vesting contract",
                    "gas_estimate": 2500000
                }
            ]
            
            for template_data in templates:
                template = ContractTemplate(
                    template_id=f"template_{secrets.token_hex(8)}",
                    name=template_data["name"],
                    contract_type=template_data["contract_type"],
                    description=template_data["description"],
                    source_code=self._generate_contract_source(template_data["contract_type"]),
                    abi=self._generate_contract_abi(template_data["contract_type"]),
                    bytecode=f"0x{secrets.token_hex(1000)}",  # Mock bytecode
                    constructor_params=self._get_constructor_params(template_data["contract_type"]),
                    functions=self._get_contract_functions(template_data["contract_type"]),
                    events=self._get_contract_events(template_data["contract_type"]),
                    gas_estimate=template_data["gas_estimate"]
                )
                
                self.templates[template.template_id] = template
            
            logger.info(f"Initialized {len(self.templates)} contract templates")
            
        except Exception as e:
            logger.error(f"Error initializing contract templates: {e}")

    def _generate_contract_source(self, contract_type: ContractType) -> str:
        """Generate contract source code"""
        # Simplified contract source code for demonstration
        sources = {
            ContractType.ERC20: """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ERC20Token {
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    
    uint256 private _totalSupply;
    string private _name;
    string private _symbol;
    uint8 private _decimals;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    constructor(string memory name_, string memory symbol_, uint8 decimals_, uint256 totalSupply_) {
        _name = name_;
        _symbol = symbol_;
        _decimals = decimals_;
        _totalSupply = totalSupply_;
        _balances[msg.sender] = totalSupply_;
    }
    
    function transfer(address to, uint256 amount) public returns (bool) {
        _transfer(msg.sender, to, amount);
        return true;
    }
    
    function approve(address spender, uint256 amount) public returns (bool) {
        _approve(msg.sender, spender, amount);
        return true;
    }
    
    function balanceOf(address account) public view returns (uint256) {
        return _balances[account];
    }
    
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }
    
    function _transfer(address from, address to, uint256 amount) internal {
        require(from != address(0), "ERC20: transfer from the zero address");
        require(to != address(0), "ERC20: transfer to the zero address");
        
        _balances[from] -= amount;
        _balances[to] += amount;
        
        emit Transfer(from, to, amount);
    }
    
    function _approve(address owner, address spender, uint256 amount) internal {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        
        _allowances[owner][spender] = amount;
        emit Approval(owner, spender, amount);
    }
}
""",
            ContractType.ERC721: """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ERC721NFT {
    mapping(uint256 => address) private _owners;
    mapping(address => uint256) private _balances;
    mapping(uint256 => address) private _tokenApprovals;
    mapping(address => mapping(address => bool)) private _operatorApprovals;
    
    string private _name;
    string private _symbol;
    uint256 private _tokenId;
    
    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
    event Approval(address indexed owner, address indexed approved, uint256 indexed tokenId);
    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);
    
    constructor(string memory name_, string memory symbol_) {
        _name = name_;
        _symbol = symbol_;
    }
    
    function mint(address to) public {
        _tokenId++;
        _mint(to, _tokenId);
    }
    
    function transferFrom(address from, address to, uint256 tokenId) public {
        require(_isApprovedOrOwner(msg.sender, tokenId), "ERC721: transfer caller is not owner nor approved");
        _transfer(from, to, tokenId);
    }
    
    function approve(address to, uint256 tokenId) public {
        address owner = ownerOf(tokenId);
        require(to != owner, "ERC721: approval to current owner");
        require(msg.sender == owner || isApprovedForAll(owner, msg.sender), "ERC721: approve caller is not owner nor approved for all");
        
        _approve(to, tokenId);
    }
    
    function ownerOf(uint256 tokenId) public view returns (address) {
        address owner = _owners[tokenId];
        require(owner != address(0), "ERC721: owner query for nonexistent token");
        return owner;
    }
    
    function _mint(address to, uint256 tokenId) internal {
        require(to != address(0), "ERC721: mint to the zero address");
        require(!_exists(tokenId), "ERC721: token already minted");
        
        _balances[to] += 1;
        _owners[tokenId] = to;
        
        emit Transfer(address(0), to, tokenId);
    }
    
    function _transfer(address from, address to, uint256 tokenId) internal {
        require(ownerOf(tokenId) == from, "ERC721: transfer of token that is not own");
        require(to != address(0), "ERC721: transfer to the zero address");
        
        _approve(address(0), tokenId);
        _balances[from] -= 1;
        _balances[to] += 1;
        _owners[tokenId] = to;
        
        emit Transfer(from, to, tokenId);
    }
    
    function _approve(address to, uint256 tokenId) internal {
        _tokenApprovals[tokenId] = to;
        emit Approval(ownerOf(tokenId), to, tokenId);
    }
    
    function _exists(uint256 tokenId) internal view returns (bool) {
        return _owners[tokenId] != address(0);
    }
    
    function _isApprovedOrOwner(address spender, uint256 tokenId) internal view returns (bool) {
        require(_exists(tokenId), "ERC721: operator query for nonexistent token");
        address owner = ownerOf(tokenId);
        return (spender == owner || getApproved(tokenId) == spender || isApprovedForAll(owner, spender));
    }
    
    function getApproved(uint256 tokenId) public view returns (address) {
        require(_exists(tokenId), "ERC721: approved query for nonexistent token");
        return _tokenApprovals[tokenId];
    }
    
    function isApprovedForAll(address owner, address operator) public view returns (bool) {
        return _operatorApprovals[owner][operator];
    }
}
"""
        }
        
        return sources.get(contract_type, "// Contract source code not available")

    def _generate_contract_abi(self, contract_type: ContractType) -> Dict[str, Any]:
        """Generate contract ABI"""
        abis = {
            ContractType.ERC20: {
                "functions": [
                    {"name": "transfer", "type": "function", "inputs": [{"name": "to", "type": "address"}, {"name": "amount", "type": "uint256"}]},
                    {"name": "approve", "type": "function", "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}]},
                    {"name": "balanceOf", "type": "function", "inputs": [{"name": "account", "type": "address"}]},
                    {"name": "totalSupply", "type": "function", "inputs": []}
                ],
                "events": [
                    {"name": "Transfer", "type": "event", "inputs": [{"name": "from", "type": "address"}, {"name": "to", "type": "address"}, {"name": "value", "type": "uint256"}]},
                    {"name": "Approval", "type": "event", "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}, {"name": "value", "type": "uint256"}]}
                ]
            },
            ContractType.ERC721: {
                "functions": [
                    {"name": "mint", "type": "function", "inputs": [{"name": "to", "type": "address"}]},
                    {"name": "transferFrom", "type": "function", "inputs": [{"name": "from", "type": "address"}, {"name": "to", "type": "address"}, {"name": "tokenId", "type": "uint256"}]},
                    {"name": "approve", "type": "function", "inputs": [{"name": "to", "type": "address"}, {"name": "tokenId", "type": "uint256"}]},
                    {"name": "ownerOf", "type": "function", "inputs": [{"name": "tokenId", "type": "uint256"}]}
                ],
                "events": [
                    {"name": "Transfer", "type": "event", "inputs": [{"name": "from", "type": "address"}, {"name": "to", "type": "address"}, {"name": "tokenId", "type": "uint256"}]},
                    {"name": "Approval", "type": "event", "inputs": [{"name": "owner", "type": "address"}, {"name": "approved", "type": "address"}, {"name": "tokenId", "type": "uint256"}]}
                ]
            }
        }
        
        return abis.get(contract_type, {})

    def _get_constructor_params(self, contract_type: ContractType) -> List[Dict[str, Any]]:
        """Get constructor parameters"""
        params = {
            ContractType.ERC20: [
                {"name": "name", "type": "string", "description": "Token name"},
                {"name": "symbol", "type": "string", "description": "Token symbol"},
                {"name": "decimals", "type": "uint8", "description": "Token decimals"},
                {"name": "totalSupply", "type": "uint256", "description": "Total supply"}
            ],
            ContractType.ERC721: [
                {"name": "name", "type": "string", "description": "NFT name"},
                {"name": "symbol", "type": "string", "description": "NFT symbol"}
            ]
        }
        
        return params.get(contract_type, [])

    def _get_contract_functions(self, contract_type: ContractType) -> List[Dict[str, Any]]:
        """Get contract functions"""
        functions = {
            ContractType.ERC20: [
                {"name": "transfer", "description": "Transfer tokens"},
                {"name": "approve", "description": "Approve spender"},
                {"name": "balanceOf", "description": "Get balance"},
                {"name": "totalSupply", "description": "Get total supply"}
            ],
            ContractType.ERC721: [
                {"name": "mint", "description": "Mint new NFT"},
                {"name": "transferFrom", "description": "Transfer NFT"},
                {"name": "approve", "description": "Approve NFT transfer"},
                {"name": "ownerOf", "description": "Get NFT owner"}
            ]
        }
        
        return functions.get(contract_type, [])

    def _get_contract_events(self, contract_type: ContractType) -> List[Dict[str, Any]]:
        """Get contract events"""
        events = {
            ContractType.ERC20: [
                {"name": "Transfer", "description": "Token transfer event"},
                {"name": "Approval", "description": "Approval event"}
            ],
            ContractType.ERC721: [
                {"name": "Transfer", "description": "NFT transfer event"},
                {"name": "Approval", "description": "NFT approval event"}
            ]
        }
        
        return events.get(contract_type, [])

    async def _initialize_existing_contracts(self):
        """Initialize existing contracts"""
        try:
            # Generate mock existing contracts
            contract_types = list(ContractType)
            blockchains = ["ethereum", "polygon", "bsc", "avalanche"]
            
            for i in range(20):  # Generate 20 mock contracts
                contract = SmartContract(
                    contract_id=f"contract_{secrets.token_hex(8)}",
                    name=f"Contract {i+1}",
                    contract_type=secrets.choice(contract_types),
                    blockchain=secrets.choice(blockchains),
                    address=f"0x{secrets.token_hex(20)}",
                    abi=self._generate_contract_abi(secrets.choice(contract_types)),
                    bytecode=f"0x{secrets.token_hex(1000)}",
                    source_code="// Contract source code",
                    compiler_version="0.8.19",
                    status=ContractStatus.DEPLOYED,
                    deployed_at=datetime.now() - timedelta(days=secrets.randbelow(365)),
                    gas_used=secrets.randbelow(2000000) + 1000000,
                    gas_price=Decimal(str(secrets.randbelow(100) + 10)),
                    deployer_address=f"0x{secrets.token_hex(20)}",
                    constructor_args=[],
                    functions=self._get_contract_functions(secrets.choice(contract_types)),
                    events=self._get_contract_events(secrets.choice(contract_types)),
                    is_verified=secrets.choice([True, False])
                )
                
                self.contracts[contract.contract_id] = contract
            
            logger.info(f"Initialized {len(self.contracts)} existing contracts")
            
        except Exception as e:
            logger.error(f"Error initializing existing contracts: {e}")

    async def _contract_monitoring_loop(self):
        """Contract monitoring loop"""
        while self.is_running:
            try:
                # Monitor contract status
                await self._monitor_contracts()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in contract monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _event_processing_loop(self):
        """Event processing loop"""
        while self.is_running:
            try:
                # Process contract events
                await self._process_contract_events()
                
                await asyncio.sleep(30)  # Process every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(30)

    async def _verification_loop(self):
        """Contract verification loop"""
        while self.is_running:
            try:
                # Verify contracts
                await self._verify_contracts()
                
                await asyncio.sleep(3600)  # Verify every hour
                
            except Exception as e:
                logger.error(f"Error in verification loop: {e}")
                await asyncio.sleep(3600)

    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                # Monitor performance
                await self._monitor_performance()
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(300)

    async def _monitor_contracts(self):
        """Monitor contracts"""
        try:
            # Simulate contract monitoring
            for contract in self.contracts.values():
                # Simulate status updates
                if contract.status == ContractStatus.DEPLOYING:
                    # Simulate deployment completion
                    if secrets.choice([True, False]):  # 50% chance
                        contract.status = ContractStatus.DEPLOYED
                        logger.info(f"Contract {contract.contract_id} deployed successfully")
                
                # Simulate verification
                if contract.status == ContractStatus.DEPLOYED and not contract.is_verified:
                    if secrets.choice([True, False]):  # 50% chance
                        contract.is_verified = True
                        contract.status = ContractStatus.VERIFIED
                        logger.info(f"Contract {contract.contract_id} verified successfully")
            
        except Exception as e:
            logger.error(f"Error monitoring contracts: {e}")

    async def _process_contract_events(self):
        """Process contract events"""
        try:
            # Generate mock contract events
            num_events = secrets.randbelow(10) + 1
            
            for _ in range(num_events):
                contract = secrets.choice(list(self.contracts.values()))
                
                event = ContractEvent(
                    event_id=f"event_{secrets.token_hex(8)}",
                    contract_id=contract.contract_id,
                    event_name=secrets.choice(contract.events) if contract.events else "Transfer",
                    block_number=secrets.randbelow(1000000) + 1000000,
                    transaction_hash=f"0x{secrets.token_hex(32)}",
                    log_index=secrets.randbelow(10),
                    topics=[f"0x{secrets.token_hex(32)}"],
                    data=f"0x{secrets.token_hex(64)}",
                    decoded_data={"from": f"0x{secrets.token_hex(20)}", "to": f"0x{secrets.token_hex(20)}", "value": secrets.randbelow(1000000)},
                    timestamp=datetime.now()
                )
                
                self.events.append(event)
            
            # Keep only last 10000 events
            if len(self.events) > 10000:
                self.events = self.events[-10000:]
            
            logger.info(f"Processed {num_events} contract events")
            
        except Exception as e:
            logger.error(f"Error processing contract events: {e}")

    async def _verify_contracts(self):
        """Verify contracts"""
        try:
            # Simulate contract verification
            for contract in self.contracts.values():
                if contract.status == ContractStatus.DEPLOYED and not contract.is_verified:
                    # Simulate verification process
                    if secrets.choice([True, False]):  # 50% chance
                        contract.is_verified = True
                        contract.status = ContractStatus.VERIFIED
                        contract.verification_tx = f"0x{secrets.token_hex(32)}"
                        logger.info(f"Verified contract: {contract.contract_id}")
            
        except Exception as e:
            logger.error(f"Error verifying contracts: {e}")

    async def _monitor_performance(self):
        """Monitor performance"""
        try:
            # Calculate performance metrics
            total_contracts = len(self.contracts)
            verified_contracts = len([c for c in self.contracts.values() if c.is_verified])
            total_interactions = len(self.interactions)
            total_events = len(self.events)
            
            # Update performance stats
            self.deployment_stats["total_contracts"] = total_contracts
            self.deployment_stats["verified_contracts"] = verified_contracts
            self.interaction_stats["total_interactions"] = total_interactions
            self.interaction_stats["total_events"] = total_events
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")

    # Public API methods
    async def get_contracts(self, contract_type: Optional[ContractType] = None,
                          blockchain: Optional[str] = None,
                          status: Optional[ContractStatus] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get contracts"""
        try:
            contracts = list(self.contracts.values())
            
            # Filter by contract type
            if contract_type:
                contracts = [c for c in contracts if c.contract_type == contract_type]
            
            # Filter by blockchain
            if blockchain:
                contracts = [c for c in contracts if c.blockchain == blockchain]
            
            # Filter by status
            if status:
                contracts = [c for c in contracts if c.status == status]
            
            # Sort by deployed time (most recent first)
            contracts = sorted(contracts, key=lambda x: x.deployed_at, reverse=True)
            
            # Limit results
            contracts = contracts[:limit]
            
            return [
                {
                    "contract_id": contract.contract_id,
                    "name": contract.name,
                    "contract_type": contract.contract_type.value,
                    "blockchain": contract.blockchain,
                    "address": contract.address,
                    "status": contract.status.value,
                    "deployed_at": contract.deployed_at.isoformat(),
                    "gas_used": contract.gas_used,
                    "gas_price": float(contract.gas_price),
                    "deployer_address": contract.deployer_address,
                    "is_verified": contract.is_verified,
                    "verification_tx": contract.verification_tx,
                    "functions": contract.functions,
                    "events": contract.events,
                    "metadata": contract.metadata
                }
                for contract in contracts
            ]
            
        except Exception as e:
            logger.error(f"Error getting contracts: {e}")
            return []

    async def get_contract_templates(self) -> List[Dict[str, Any]]:
        """Get contract templates"""
        try:
            return [
                {
                    "template_id": template.template_id,
                    "name": template.name,
                    "contract_type": template.contract_type.value,
                    "description": template.description,
                    "gas_estimate": template.gas_estimate,
                    "constructor_params": template.constructor_params,
                    "functions": template.functions,
                    "events": template.events,
                    "created_at": template.created_at.isoformat(),
                    "metadata": template.metadata
                }
                for template in self.templates.values()
            ]
            
        except Exception as e:
            logger.error(f"Error getting contract templates: {e}")
            return []

    async def get_contract_events(self, contract_id: Optional[str] = None,
                                event_name: Optional[str] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Get contract events"""
        try:
            events = self.events
            
            # Filter by contract ID
            if contract_id:
                events = [e for e in events if e.contract_id == contract_id]
            
            # Filter by event name
            if event_name:
                events = [e for e in events if e.event_name == event_name]
            
            # Sort by timestamp (most recent first)
            events = sorted(events, key=lambda x: x.timestamp, reverse=True)
            
            # Limit results
            events = events[:limit]
            
            return [
                {
                    "event_id": event.event_id,
                    "contract_id": event.contract_id,
                    "event_name": event.event_name,
                    "block_number": event.block_number,
                    "transaction_hash": event.transaction_hash,
                    "log_index": event.log_index,
                    "topics": event.topics,
                    "data": event.data,
                    "decoded_data": event.decoded_data,
                    "timestamp": event.timestamp.isoformat(),
                    "metadata": event.metadata
                }
                for event in events
            ]
            
        except Exception as e:
            logger.error(f"Error getting contract events: {e}")
            return []

    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        try:
            return {
                "engine_id": self.engine_id,
                "is_running": self.is_running,
                "total_contracts": len(self.contracts),
                "total_templates": len(self.templates),
                "total_interactions": len(self.interactions),
                "total_events": len(self.events),
                "verified_contracts": len([c for c in self.contracts.values() if c.is_verified]),
                "deployment_stats": self.deployment_stats,
                "interaction_stats": self.interaction_stats,
                "supported_contract_types": [ct.value for ct in ContractType],
                "supported_blockchains": list(self.blockchain_configs.keys()),
                "uptime": "active" if self.is_running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error getting engine metrics: {e}")
            return {}

    async def deploy_contract(self, template_id: str, name: str, blockchain: str,
                            constructor_args: List[Any], deployer_address: str) -> str:
        """Deploy a contract"""
        try:
            if template_id not in self.templates:
                raise ValueError(f"Template {template_id} not found")
            
            template = self.templates[template_id]
            
            # Create contract
            contract = SmartContract(
                contract_id=f"contract_{secrets.token_hex(8)}",
                name=name,
                contract_type=template.contract_type,
                blockchain=blockchain,
                address=f"0x{secrets.token_hex(20)}",  # Mock address
                abi=template.abi,
                bytecode=template.bytecode,
                source_code=template.source_code,
                compiler_version="0.8.19",
                status=ContractStatus.DEPLOYING,
                deployed_at=datetime.now(),
                gas_used=template.gas_estimate,
                gas_price=Decimal(str(secrets.randbelow(100) + 10)),
                deployer_address=deployer_address,
                constructor_args=constructor_args,
                functions=[f["name"] for f in template.functions],
                events=[e["name"] for e in template.events]
            )
            
            self.contracts[contract.contract_id] = contract
            
            logger.info(f"Deployed contract: {contract.contract_id}")
            return contract.contract_id
            
        except Exception as e:
            logger.error(f"Error deploying contract: {e}")
            raise

    async def interact_with_contract(self, contract_id: str, function_name: str,
                                   caller_address: str, parameters: Dict[str, Any],
                                   gas_limit: int, gas_price: Decimal,
                                   value: Decimal = Decimal("0")) -> str:
        """Interact with a contract"""
        try:
            if contract_id not in self.contracts:
                raise ValueError(f"Contract {contract_id} not found")
            
            contract = self.contracts[contract_id]
            
            # Create interaction
            interaction = ContractInteraction(
                interaction_id=f"interaction_{secrets.token_hex(8)}",
                contract_id=contract_id,
                function_name=function_name,
                function_type=ContractFunction.TRANSFER,  # Simplified
                caller_address=caller_address,
                parameters=parameters,
                gas_limit=gas_limit,
                gas_price=gas_price,
                value=value,
                tx_hash=f"0x{secrets.token_hex(32)}",
                block_number=secrets.randbelow(1000000) + 1000000,
                status="confirmed",
                gas_used=secrets.randbelow(gas_limit) + 21000,
                timestamp=datetime.now()
            )
            
            self.interactions.append(interaction)
            
            # Keep only last 10000 interactions
            if len(self.interactions) > 10000:
                self.interactions = self.interactions[-10000:]
            
            logger.info(f"Contract interaction: {interaction.interaction_id}")
            return interaction.interaction_id
            
        except Exception as e:
            logger.error(f"Error interacting with contract: {e}")
            raise

# Global instance
smart_contract_engine = SmartContractEngine()
