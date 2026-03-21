import { ThemeProvider } from './ThemeContext';
import { SettingsProvider } from './SettingsContext';
import Dashboard from './components/Dashboard';

function App() {
  return (
    <ThemeProvider>
      <SettingsProvider>
        <Dashboard />
      </SettingsProvider>
    </ThemeProvider>
  );
}

export default App;
