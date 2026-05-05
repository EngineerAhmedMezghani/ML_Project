<template>
  <div class="p-8">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900">Feature Selector &amp; Churn Prediction</h1>
      <p class="text-gray-600 mt-1">
        Enter the 9 known customer features. The backend predicts all remaining features and the churn risk.
      </p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
      <!-- Input Form -->
      <div class="lg:col-span-1">
        <div class="card">
          <div class="flex items-center gap-2 mb-6">
            <User class="w-5 h-5 text-primary-600" />
            <h2 class="text-lg font-semibold text-gray-900">Customer Inputs</h2>
          </div>

          <form @submit.prevent="handlePredict" class="space-y-4">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Recency</label>
              <input v-model.number="form.Recency" type="number" class="input-field" placeholder="e.g., 302" />
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Age</label>
              <input v-model.number="form.Age" type="number" class="input-field" placeholder="e.g., 49" />
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Region</label>
              <select v-model="form.Region" class="input-field">
                <option v-for="opt in meta.Region" :key="opt" :value="opt">{{ opt }}</option>
              </select>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Monetary Total</label>
              <input v-model.number="form.MonetaryTotal" type="number" step="0.01" class="input-field" placeholder="e.g., 5288.63" />
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Frequency</label>
              <input v-model.number="form.Frequency" type="number" class="input-field" placeholder="e.g., 35" />
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Satisfaction Score</label>
              <input v-model.number="form.SatisfactionScore" type="number" min="1" max="5" class="input-field" placeholder="1–5" />
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Loyalty Level</label>
              <select v-model="form.LoyaltyLevel" class="input-field">
                <option v-for="opt in meta.LoyaltyLevel" :key="opt" :value="opt">{{ opt }}</option>
              </select>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Customer Type</label>
              <select v-model="form.CustomerType" class="input-field">
                <option v-for="opt in meta.CustomerType" :key="opt" :value="opt">{{ opt }}</option>
              </select>
            </div>

            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Account Status</label>
              <select v-model="form.AccountStatus" class="input-field">
                <option v-for="opt in meta.AccountStatus" :key="opt" :value="opt">{{ opt }}</option>
              </select>
            </div>

            <div class="flex gap-3 pt-4">
              <button type="submit" :disabled="loading" class="btn-primary flex items-center gap-2 w-full justify-center">
                <Sparkles v-if="!loading" class="w-4 h-4" />
                <Loader v-else class="w-4 h-4 animate-spin" />
                {{ loading ? 'Predicting…' : 'Predict' }}
              </button>
              <button type="button" @click="resetForm" class="btn-secondary">
                Reset
              </button>
            </div>
          </form>

          <div v-if="error" class="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {{ error }}
          </div>
        </div>
      </div>

      <!-- Results Panel -->
      <div class="lg:col-span-2 space-y-6">
        <!-- Empty State -->
        <div v-if="!result && !loading" class="card h-64 flex flex-col items-center justify-center text-center">
          <div class="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
            <Brain class="w-8 h-8 text-gray-400" />
          </div>
          <h3 class="text-lg font-medium text-gray-900 mb-2">Ready to Predict</h3>
          <p class="text-sm text-gray-500 max-w-md">
            Fill in the 9 customer features on the left and click <strong>Predict</strong> to generate all remaining features and churn risk.
          </p>
        </div>

        <!-- Loading State -->
        <div v-if="loading" class="card h-64 flex flex-col items-center justify-center text-center">
          <Loader class="w-10 h-10 text-primary-600 animate-spin mb-4" />
          <p class="text-gray-600">Running model inference…</p>
        </div>

        <!-- Churn Card -->
        <div v-if="result && result.churn" class="card" :class="churnCardClass">
          <div class="flex items-center justify-between">
            
            <div>
              <p class="text-sm font-medium text-gray-600 mb-1">Churn Probability</p>

              <!-- on affiche directement la probabilité de churn -->
              <p class="text-4xl font-bold text-red-600">
                {{ (result.churn.probability * 100).toFixed(1) }}%
              </p>

              <span class="inline-block mt-2 px-3 py-1 rounded-full text-sm font-medium" :class="riskBadgeClass">
                {{ result.churn.risk_level }} Risk
              </span>
            </div>

            <div class="text-right">
              <p class="text-sm text-gray-600">Prediction</p>
              <p class="text-xl font-semibold"
                :class="result.churn.prediction === 1 ? 'text-red-600' : 'text-green-600'">

                {{ result.churn.prediction === 1 ? 'Will Churn' : 'Will Stay' }}
              </p>
            </div>

          </div>
        </div>

        <!-- Predicted Features -->
        <div v-if="result && result.predictions" class="card">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-gray-900">Predicted Features</h3>
            <span class="text-xs text-gray-500">{{ Object.keys(result.predictions).length }} features generated</span>
          </div>

          <div class="space-y-4">
            <div v-for="(catKeys, catName) in result.categories" :key="catName">
              <h4 class="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">{{ catName }}</h4>
              <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                <div v-for="key in catKeys" :key="key" class="p-3 bg-gray-50 rounded-lg border border-gray-100">
                  <p class="text-xs text-gray-500 truncate" :title="key">{{ key }}</p>
                  <p class="text-sm font-semibold text-gray-900 truncate" :title="String(result.predictions[key])">
                    {{ formatValue(result.predictions[key]) }}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { User, Sparkles, Brain, Loader } from 'lucide-vue-next'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

const form = ref({
  Recency: 302,
  Age: 49,
  Region: 'UK',
  MonetaryTotal: 5288.63,
  Frequency: 35,
  SatisfactionScore: 4,
  LoyaltyLevel: 'Jeune',
  CustomerType: 'Perdu',
  AccountStatus: 'Active',
})

const meta = ref({
  Region: ['UK', 'Europe continentale', 'Océanie', 'Europe du Nord', 'Autre', 'Europe centrale', "Europe de l'Est", 'Asie', 'Moyen-Orient', 'Amérique du Nord', 'Amérique du Sud', 'Afrique'],
  LoyaltyLevel: ['Nouveau', 'Jeune', 'Établi', 'Ancien'],
  CustomerType: ['Hyperactif', 'Nouveau', 'Occasionnel', 'Perdu', 'Régulier'],
  AccountStatus: ['Active', 'Inactive'],
})

const loading = ref(false)
const error = ref('')
const result = ref(null)

onMounted(async () => {
  try {
    const res = await fetch(`${API_BASE}/inputs`)
    if (res.ok) {
      const data = await res.json()
      // merge server metadata for categorical options if available
      data.inputs.forEach(inp => {
        if (inp.type === 'categorical' && inp.options) {
          meta.value[inp.name] = inp.options
        }
      })
    }
  } catch (e) {
    // fallback to hardcoded meta
  }
})

async function handlePredict () {
  loading.value = true
  error.value = ''
  result.value = null

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form.value),
    })
    const data = await res.json()
    if (!res.ok) {
      throw new Error(data.error || 'Prediction failed')
    }
    result.value = data
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

function resetForm () {
  form.value = {
    Recency: null,
    Age: null,
    Region: 'UK',
    MonetaryTotal: null,
    Frequency: null,
    SatisfactionScore: null,
    LoyaltyLevel: 'Jeune',
    CustomerType: 'Perdu',
    AccountStatus: 'Active',
  }
  result.value = null
  error.value = ''
}

function formatValue (v) {
  if (typeof v === 'number') {
    return Number.isInteger(v) ? v : v.toFixed(2)
  }
  return String(v)
}

const churnCardClass = computed(() => {
  if (!result.value?.churn) return ''
  const p = result.value.churn.probability
  if (p > 0.7) return 'border-red-200 bg-red-50'
  if (p > 0.3) return 'border-yellow-200 bg-yellow-50'
  return 'border-green-200 bg-green-50'
})

const churnTextClass = computed(() => {
  if (!result.value?.churn) return ''
  const p = result.value.churn.probability
  if (p > 0.7) return 'text-red-600'
  if (p > 0.3) return 'text-yellow-600'
  return 'text-green-600'
})

const riskBadgeClass = computed(() => {
  if (!result.value?.churn) return ''
  const level = result.value.churn.risk_level
  if (level === 'High') return 'bg-red-100 text-red-700'
  if (level === 'Medium') return 'bg-yellow-100 text-yellow-700'
  return 'bg-green-100 text-green-700'
})
</script>
