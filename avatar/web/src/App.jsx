import { useState } from 'react'
import VisemePlayer from './components/VisemePlayer.jsx'
import PhoneticsView from './components/PhoneticsView.jsx'

// Speech-to-visual prototypes. Two views behind one app:
//   1) Player    - PoC3 PixiJS lip-sync player (audio clock -> swaps pose sprites)
//   2) Phonetics - PoC2 alignment strip (faster-whisper word timings -> viseme timeline)
const TABS = [
  { id: 'player', label: 'Avatar · Lip-Sync' },
  { id: 'phonetics', label: 'Phonetics · Alignment' },
]

export default function App() {
  const [tab, setTab] = useState('player')

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '12px',
        padding: '16px',
      }}
    >
      <h2
        style={{
          color: '#e0e0e0',
          fontFamily: 'sans-serif',
          fontSize: '14px',
          letterSpacing: '1px',
          textTransform: 'uppercase',
          margin: 0,
        }}
      >
        Speech → Visual Prototypes
      </h2>

      {/* sub-nav between the two prototypes */}
      <div style={{ display: 'flex', gap: '8px' }}>
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              fontFamily: 'sans-serif',
              fontSize: '13px',
              padding: '7px 14px',
              borderRadius: '8px',
              cursor: 'pointer',
              border: tab === t.id ? 'none' : '1px solid #2a313b',
              background: tab === t.id ? '#1f6feb' : '#11161d',
              color: tab === t.id ? '#fff' : '#e6edf3',
              fontWeight: tab === t.id ? 600 : 400,
            }}
          >
            {t.label}
          </button>
        ))}
        <a
          href="/"
          style={{
            fontFamily: 'sans-serif',
            fontSize: '13px',
            padding: '7px 14px',
            borderRadius: '8px',
            border: '1px solid #2a313b',
            background: '#11161d',
            color: '#e6edf3',
            textDecoration: 'none',
          }}
        >
          ◂ Back to demos
        </a>
      </div>

      {tab === 'player' ? <VisemePlayer /> : <PhoneticsView />}
    </div>
  )
}
