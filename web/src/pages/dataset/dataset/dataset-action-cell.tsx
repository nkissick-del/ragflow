import { ConfirmDeleteDialog } from '@/components/confirm-delete-dialog';
import { Button } from '@/components/ui/button';
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from '@/components/ui/hover-card';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { DocumentType } from '@/constants/knowledge';
import { useRemoveDocument } from '@/hooks/use-document-request';
import { IDocumentInfo } from '@/interfaces/database/document';
import { formatFileSize } from '@/utils/common-util';
import { formatDate } from '@/utils/date';
import { downloadDocument } from '@/utils/file-util';
import { Download, Eye, PenLine, Trash2 } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { UseRenameDocumentShowType } from './use-rename-document';
import { isParserRunning } from './utils';

const Fields = ['name', 'size', 'type', 'create_time', 'update_time'];

const FunctionMap = {
  size: formatFileSize,
  create_time: formatDate,
  update_time: formatDate,
};

export function DatasetActionCell({
  record,
  showRenameModal,
}: { record: IDocumentInfo } & UseRenameDocumentShowType) {
  const { id, run, type } = record;
  const isRunning = isParserRunning(run);
  const isVirtualDocument = type === DocumentType.Virtual;
  const { t } = useTranslation();

  const { removeDocument } = useRemoveDocument();

  const onDownloadDocument = useCallback(() => {
    downloadDocument({
      id,
      filename: record.name,
    });
  }, [id, record.name]);

  const handleRemove = useCallback(() => {
    removeDocument(id);
  }, [id, removeDocument]);

  const handleRename = useCallback(() => {
    showRenameModal(record);
  }, [record, showRenameModal]);

  return (
    <section className="flex gap-4 items-center text-text-sub-title-invert opacity-0 group-hover:opacity-100 transition-opacity">
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="transparent"
            className="border-none hover:bg-bg-card text-text-primary"
            size={'sm'}
            disabled={isRunning}
            onClick={handleRename}
            aria-label={t('common.rename')}
          >
            <PenLine />
          </Button>
        </TooltipTrigger>
        <TooltipContent>{t('common.rename')}</TooltipContent>
      </Tooltip>
      <HoverCard>
        <HoverCardTrigger>
          <Button
            variant="transparent"
            className="border-none hover:bg-bg-card text-text-primary"
            disabled={isRunning}
            size={'sm'}
            aria-label={t('knowledgeDetails.view')}
          >
            <Eye />
          </Button>
        </HoverCardTrigger>
        <HoverCardContent className="w-[40vw] max-h-[40vh] overflow-auto">
          <ul className="space-y-2">
            {Object.entries(record)
              .filter(([key]) => Fields.some((x) => x === key))

              .map(([key, value], idx) => {
                return (
                  <li key={idx} className="flex gap-2">
                    {key}:
                    <div>
                      {key in FunctionMap
                        ? FunctionMap[key as keyof typeof FunctionMap](value)
                        : value}
                    </div>
                  </li>
                );
              })}
          </ul>
        </HoverCardContent>
      </HoverCard>

      {isVirtualDocument || (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="transparent"
              className="border-none hover:bg-bg-card text-text-primary"
              onClick={onDownloadDocument}
              disabled={isRunning}
              size={'sm'}
              aria-label={t('common.download')}
            >
              <Download />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('common.download')}</TooltipContent>
        </Tooltip>
      )}
      <Tooltip>
        <ConfirmDeleteDialog onOk={handleRemove}>
          <TooltipTrigger asChild>
            <Button
              variant="transparent"
              className="border-none hover:bg-bg-card text-text-primary"
              size={'sm'}
              disabled={isRunning}
              aria-label={t('common.delete')}
            >
              <Trash2 />
            </Button>
          </TooltipTrigger>
        </ConfirmDeleteDialog>
        <TooltipContent>{t('common.delete')}</TooltipContent>
      </Tooltip>
    </section>
  );
}
