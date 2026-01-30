import { useTranslate } from '@/hooks/common-hooks';
import { CheckOutlined, CopyOutlined } from '@ant-design/icons';
import { useState } from 'react';
import { CopyToClipboard as Clipboard, Props } from 'react-copy-to-clipboard';
import { Tooltip, TooltipContent, TooltipTrigger } from './ui/tooltip';

const CopyToClipboard = ({ text }: Props) => {
  const [copied, setCopied] = useState(false);
  const { t } = useTranslate('common');

  const handleCopy = () => {
    setCopied(true);
    setTimeout(() => {
      setCopied(false);
    }, 2000);
  };

  return (
    <Tooltip>
      <Clipboard text={text} onCopy={handleCopy}>
        <TooltipTrigger asChild>
          <button
            aria-label={t('copy')}
            className="flex cursor-pointer items-center border-none bg-transparent p-0 outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2"
            type="button"
          >
            {copied ? <CheckOutlined /> : <CopyOutlined />}
          </button>
        </TooltipTrigger>
      </Clipboard>
      <TooltipContent>{copied ? t('copied') : t('copy')}</TooltipContent>
    </Tooltip>
  );
};

export default CopyToClipboard;

export function CopyToClipboardWithText({ text }: { text: string }) {
  return (
    <div className="bg-bg-card p-1 rounded-md flex gap-2">
      <span className="flex-1 truncate">{text}</span>
      <CopyToClipboard text={text}></CopyToClipboard>
    </div>
  );
}
