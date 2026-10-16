import axios from 'axios'
import type { QueryResponse, EmendaData, PaginatedResponse } from '../types'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
})

// Pipeline RAG encadeia 3 chamadas ao LLM local (interpreter -> planner ->
// synthesizer); em CPU isso leva minutos. O timeout da consulta precisa de uma
// folga grande, enquanto os GETs rapidos mantem um teto curto.
const QUERY_TIMEOUT_MS = 600000 // 10 min
const FAST_TIMEOUT_MS = 30000

export async function enviarConsulta(consulta: string): Promise<QueryResponse> {
  try {
    const { data } = await api.post<QueryResponse>('/query', { consulta }, { timeout: QUERY_TIMEOUT_MS })
    return data
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.data?.detail) {
      throw new Error(err.response.data.detail)
    }
    throw err
  }
}

export async function buscarEmendas(
  params: Record<string, string | number>
): Promise<PaginatedResponse<EmendaData>> {
  const { data } = await api.get('/emendas', { params, timeout: FAST_TIMEOUT_MS })
  return data
}

export async function checkHealth() {
  const { data } = await api.get('/health', { timeout: FAST_TIMEOUT_MS })
  return data
}
