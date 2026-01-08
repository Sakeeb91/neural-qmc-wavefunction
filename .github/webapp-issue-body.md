## Summary

Build an interactive web application using **FastAPI** (backend) and **React** (frontend) to visualize Neural QMC simulations, monitor training in real-time, and compare results against classical quantum chemistry methods.

**Parent Issue:** #1

## Dependencies

| Depends On | Reason |
|------------|--------|
| #2 (Phase 1) | Need basic VMC loop and Molecule class |
| #3 (Phase 2) | Need antisymmetric wavefunction for proper visualization |
| #5 (Phase 4) | Need H₄ implementation for meaningful demos |
| #6 (Phase 5) | Need PySCF integration for method comparisons |

**Can Start After:** Phase 2 complete (basic visualizations)
**Full Features After:** Phase 5 complete (all comparisons)

## Objectives

- Create FastAPI backend serving Neural QMC computations
- Build React frontend with interactive 3D molecular visualizations
- Real-time training progress with WebSocket updates
- Interactive molecule builder and parameter tuning
- Publication-ready figure export

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEURAL QMC WEBAPP                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         REACT FRONTEND                               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│  │  │  Molecule   │  │  Training   │  │  Results    │  │  Compare   │  │    │
│  │  │  Builder    │  │  Dashboard  │  │  Viewer     │  │  Methods   │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │    │
│  │         │                │                │               │          │    │
│  │         └────────────────┴────────────────┴───────────────┘          │    │
│  │                                   │                                   │    │
│  │                          WebSocket + REST API                         │    │
│  └───────────────────────────────────┬───────────────────────────────────┘    │
│                                      │                                        │
│  ┌───────────────────────────────────┴───────────────────────────────────┐    │
│  │                         FASTAPI BACKEND                                │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐    │    │
│  │  │  /api/      │  │  /api/      │  │  /api/      │  │  /ws/      │    │    │
│  │  │  molecules  │  │  train      │  │  compare    │  │  training  │    │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘    │    │
│  │         │                │                │               │            │    │
│  │         └────────────────┴────────────────┴───────────────┘            │    │
│  │                                   │                                     │    │
│  │                          NQMC Core Library                              │    │
│  └───────────────────────────────────┴─────────────────────────────────────┘    │
│                                      │                                          │
│  ┌───────────────────────────────────┴───────────────────────────────────┐      │
│  │                         COMPUTATION LAYER                              │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐    │      │
│  │  │  Neural     │  │  MCMC       │  │  VMC        │  │  PySCF     │    │      │
│  │  │  Wavefunction│  │  Sampler   │  │  Optimizer  │  │  Baselines │    │      │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘    │      │
│  └────────────────────────────────────────────────────────────────────────┘      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Features

### 1. Molecule Builder
- Visual molecule constructor (drag atoms, set bond lengths)
- Preset molecules: H₂, H₄, H₆, LiH
- Custom hydrogen chain generator
- Real-time geometry validation

### 2. Training Dashboard
- **Real-time energy plot** via WebSocket
- MCMC acceptance rate monitor
- Local energy variance tracker
- Parameter gradient norms
- Pause/resume/stop controls
- Checkpoint save/load

### 3. Wavefunction Visualizer
- **3D electron density** isosurfaces (Three.js/React Three Fiber)
- 2D slices through molecular plane
- Orbital visualization
- Animated MCMC walker trajectories

### 4. Method Comparison
- Side-by-side: Neural QMC vs HF vs MP2 vs CCSD
- Interactive PES curves (click to see geometry)
- Correlation energy breakdown
- Error bar visualization
- Export publication figures (PNG, SVG, PDF)

### 5. Results Export
- Download trained model checkpoints
- Export energy data (CSV, JSON)
- Generate LaTeX tables
- Create shareable visualization links

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Backend Framework** | FastAPI | Async, fast, auto OpenAPI docs |
| **WebSocket** | FastAPI WebSocket | Real-time training updates |
| **Task Queue** | Celery + Redis | Long-running training jobs |
| **Frontend Framework** | React 18 + TypeScript | Modern, typed, ecosystem |
| **State Management** | Zustand | Lightweight, simple |
| **UI Components** | shadcn/ui + Tailwind | Beautiful, accessible |
| **Charts** | Recharts | React-native charting |
| **3D Visualization** | React Three Fiber | Three.js for React |
| **API Client** | TanStack Query | Caching, sync, WebSocket |
| **Build Tool** | Vite | Fast dev server |

## Directory Structure

```
webapp/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app entry
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── molecules.py    # Molecule CRUD
│   │   │   │   ├── training.py     # Training endpoints
│   │   │   │   ├── results.py      # Results retrieval
│   │   │   │   └── comparison.py   # Method comparison
│   │   │   └── websocket.py        # WebSocket handlers
│   │   ├── core/
│   │   │   ├── config.py           # Settings
│   │   │   └── nqmc_bridge.py      # Bridge to NQMC library
│   │   ├── models/
│   │   │   └── schemas.py          # Pydantic models
│   │   └── services/
│   │       ├── training.py         # Training service
│   │       └── visualization.py    # Plot generation
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── MoleculeBuilder/
│   │   │   ├── TrainingDashboard/
│   │   │   ├── WavefunctionViewer/
│   │   │   ├── MethodComparison/
│   │   │   └── ui/                 # shadcn components
│   │   ├── hooks/
│   │   │   ├── useTraining.ts
│   │   │   ├── useWebSocket.ts
│   │   │   └── useMolecule.ts
│   │   ├── stores/
│   │   │   └── appStore.ts
│   │   ├── api/
│   │   │   └── client.ts
│   │   ├── pages/
│   │   │   ├── Home.tsx
│   │   │   ├── Train.tsx
│   │   │   ├── Visualize.tsx
│   │   │   └── Compare.tsx
│   │   └── App.tsx
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## Implementation Tasks

### Backend Tasks

- [ ] **Setup FastAPI project structure**
  - Initialize with Poetry/pip
  - Configure CORS for frontend
  - Set up OpenAPI documentation

- [ ] **Create molecule endpoints**
  ```python
  @router.post("/molecules")
  async def create_molecule(config: MoleculeConfig) -> MoleculeResponse:
      """Create a new molecular system."""
      molecule = Molecule.from_atoms(config.elements, config.positions)
      return MoleculeResponse(
          id=uuid4(),
          n_electrons=molecule.n_electrons,
          n_atoms=molecule.n_atoms,
          nuclear_repulsion=molecule.nuclear_repulsion_energy()
      )
  ```

- [ ] **Create training endpoints with WebSocket**
  ```python
  @router.post("/training/start")
  async def start_training(config: TrainingConfig) -> TrainingJob:
      """Start a VMC training job."""
      job_id = await training_service.start(config)
      return TrainingJob(id=job_id, status="running")

  @router.websocket("/ws/training/{job_id}")
  async def training_websocket(websocket: WebSocket, job_id: str):
      """Stream training metrics in real-time."""
      await websocket.accept()
      async for update in training_service.stream(job_id):
          await websocket.send_json(update.dict())
  ```

- [ ] **Create comparison endpoints**
  ```python
  @router.post("/compare")
  async def compare_methods(config: ComparisonConfig) -> ComparisonResult:
      """Run Neural QMC and PySCF methods for comparison."""
      nqmc_result = await run_nqmc(config.molecule, config.nqmc_params)
      pyscf_results = await run_pyscf(config.molecule, config.methods)
      return ComparisonResult(nqmc=nqmc_result, classical=pyscf_results)
  ```

- [ ] **Implement visualization data endpoints**
  ```python
  @router.get("/visualization/density/{job_id}")
  async def get_electron_density(job_id: str, resolution: int = 50):
      """Get electron density grid for 3D visualization."""
      return await visualization_service.compute_density_grid(job_id, resolution)
  ```

### Frontend Tasks

- [ ] **Set up React + Vite + TypeScript project**
- [ ] **Install and configure shadcn/ui components**
- [ ] **Create MoleculeBuilder component**
  ```tsx
  const MoleculeBuilder: React.FC = () => {
    const [atoms, setAtoms] = useState<Atom[]>([]);
    const [bondLength, setBondLength] = useState(1.4);

    const presets = {
      H2: () => createHydrogenMolecule(bondLength),
      H4: () => createHydrogenChain(4, bondLength),
      H6: () => createHydrogenChain(6, bondLength),
    };

    return (
      <Card>
        <CardHeader>
          <CardTitle>Molecule Builder</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2 mb-4">
            {Object.entries(presets).map(([name, fn]) => (
              <Button key={name} onClick={() => setAtoms(fn())}>
                {name}
              </Button>
            ))}
          </div>
          <Slider
            label="Bond Length (Bohr)"
            value={bondLength}
            onChange={setBondLength}
            min={0.5}
            max={4.0}
            step={0.1}
          />
          <MoleculeViewer3D atoms={atoms} />
        </CardContent>
      </Card>
    );
  };
  ```

- [ ] **Create TrainingDashboard with real-time updates**
  ```tsx
  const TrainingDashboard: React.FC<{ jobId: string }> = ({ jobId }) => {
    const { energies, variance, acceptanceRate, isConnected } = useTrainingWebSocket(jobId);

    return (
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Energy Convergence</CardTitle>
            <Badge variant={isConnected ? "success" : "destructive"}>
              {isConnected ? "Live" : "Disconnected"}
            </Badge>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={energies}>
                <XAxis dataKey="step" />
                <YAxis domain={['auto', 'auto']} />
                <Line type="monotone" dataKey="energy" stroke="#3b82f6" dot={false} />
                <Line type="monotone" dataKey="target" stroke="#ef4444" strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>MCMC Diagnostics</CardTitle></CardHeader>
          <CardContent>
            <div className="space-y-4">
              <Metric label="Acceptance Rate" value={acceptanceRate} target={0.5} />
              <Metric label="Energy Variance" value={variance} />
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };
  ```

- [ ] **Create 3D WavefunctionViewer with React Three Fiber**
  ```tsx
  const WavefunctionViewer: React.FC<{ densityData: DensityGrid }> = ({ densityData }) => {
    return (
      <Canvas camera={{ position: [5, 5, 5] }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />

        {/* Electron density isosurface */}
        <IsoSurface data={densityData} isovalue={0.05} color="#3b82f6" opacity={0.6} />

        {/* Nuclear positions */}
        {densityData.nuclei.map((nucleus, i) => (
          <Sphere key={i} position={nucleus.position} args={[0.2]}>
            <meshStandardMaterial color="red" />
          </Sphere>
        ))}

        <OrbitControls />
        <axesHelper args={[3]} />
      </Canvas>
    );
  };
  ```

- [ ] **Create MethodComparison component**
  ```tsx
  const MethodComparison: React.FC = () => {
    const { data, isLoading } = useQuery(['comparison'], fetchComparison);

    if (isLoading) return <Skeleton className="h-96" />;

    return (
      <Card>
        <CardHeader>
          <CardTitle>Method Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={data.methods}>
              <XAxis dataKey="name" />
              <YAxis label={{ value: 'Energy (Ha)', angle: -90 }} />
              <Bar dataKey="energy" fill="#3b82f6">
                {data.methods.map((entry, index) => (
                  <Cell key={index} fill={METHOD_COLORS[entry.name]} />
                ))}
              </Bar>
              <ErrorBar dataKey="error" />
              <ReferenceLine y={data.exact} stroke="red" strokeDasharray="3 3" />
            </BarChart>
          </ResponsiveContainer>

          <Table className="mt-4">
            <TableHeader>
              <TableRow>
                <TableHead>Method</TableHead>
                <TableHead>Energy (Ha)</TableHead>
                <TableHead>Error (mHa)</TableHead>
                <TableHead>Correlation %</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.methods.map((method) => (
                <TableRow key={method.name}>
                  <TableCell>{method.name}</TableCell>
                  <TableCell>{method.energy.toFixed(6)}</TableCell>
                  <TableCell>±{(method.error * 1000).toFixed(2)}</TableCell>
                  <TableCell>{method.correlationPct.toFixed(1)}%</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    );
  };
  ```

### Integration Tasks

- [ ] **Set up Docker Compose for full stack**
  ```yaml
  version: '3.8'
  services:
    backend:
      build: ./backend
      ports:
        - "8000:8000"
      volumes:
        - ../src:/app/nqmc  # Mount NQMC library
      environment:
        - REDIS_URL=redis://redis:6379

    frontend:
      build: ./frontend
      ports:
        - "3000:3000"
      depends_on:
        - backend

    redis:
      image: redis:alpine
      ports:
        - "6379:6379"
  ```

- [ ] **Create API client with TanStack Query**
- [ ] **Implement WebSocket reconnection logic**
- [ ] **Add export functionality (PNG, SVG, CSV)**

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/molecules` | Create molecular system |
| GET | `/api/molecules/{id}` | Get molecule details |
| GET | `/api/molecules/presets` | List preset molecules |
| POST | `/api/training/start` | Start VMC training |
| GET | `/api/training/{id}/status` | Get training status |
| POST | `/api/training/{id}/stop` | Stop training |
| WS | `/ws/training/{id}` | Real-time training updates |
| GET | `/api/results/{id}` | Get training results |
| GET | `/api/visualization/density/{id}` | Get density grid |
| POST | `/api/compare` | Compare methods |
| GET | `/api/export/{id}/{format}` | Export results |

## Definition of Done

- [ ] Backend API fully functional with OpenAPI docs
- [ ] Frontend renders all major components
- [ ] Real-time training updates work via WebSocket
- [ ] 3D electron density visualization renders correctly
- [ ] Method comparison chart displays accurate data
- [ ] Docker Compose brings up full stack
- [ ] README with setup instructions
- [ ] Demo video/GIF for repository

## Screenshots (Planned)

| View | Description |
|------|-------------|
| Home | Landing page with molecule presets |
| Train | Real-time training dashboard |
| Visualize | 3D electron density viewer |
| Compare | Method comparison bar chart |

## Performance Considerations

- Use WebSocket for training updates (not polling)
- Compute density grids on backend, send compressed
- Lazy load 3D components (React.lazy)
- Cache PySCF results (they don't change)
- Use Web Workers for heavy frontend computation
